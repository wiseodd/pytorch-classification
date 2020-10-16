import argparse
import os
import shutil
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models.cifar as models
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', default='cifar10', type=str)
parser.add_argument('--mcd', default=False, action='store_true')
parser.add_argument('--deepens', default=False, action='store_true')
args = parser.parse_args()

num_classes = 100 if args.dataset == 'cifar100' else 10


def get_model(fname: str):
    model = torch.nn.DataParallel(models.densenetbc121(num_classes=num_classes))
    checkpoint = torch.load(fname)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    # Translate last-layers classifier
    state_dict['module.clf.0.weight'] = state_dict.pop('module.conv2.weight')
    state_dict['module.clf.0.bias'] = state_dict.pop('module.conv2.bias')
    state_dict['module.clf.4.weight'] = state_dict.pop('module.fc.weight')
    state_dict['module.clf.4.bias'] = state_dict.pop('module.fc.bias')

    # Unwrap DataParallel
    state_dict_noDP = {}

    for k in state_dict.keys():
        # Remove "module."
        state_dict_noDP[k[7:]] = state_dict[k]

    # Check load on non-DataParallel model
    model2 = models.densenetbc121_Alt(num_classes=num_classes).cuda()
    model2.load_state_dict(state_dict_noDP)

    return model2


path_plain = f'checkpoints/{args.dataset}'

if args.mcd:
    path = path_plain + '/dropout'
elif args.deepens:
    path = path_plain + '/deepens'
else:
    path = path_plain


if not args.deepens:  # Plain and dropout
    model = get_model(f'{path}/densenet-bc-100-12/model_best.pt')
    type = 'mcd' if args.mcd else 'plain'
    torch.save(model.state_dict(), f'checkpoints/{args.dataset.upper()}_{type}_densenet.pt')
else:
    # The first model is the "plain" model
    models_de = [get_model(f'{path_plain}/densenet-bc-100-12/model_best.pt')]

    for i in range(1, 5):  # 1,...,4
        model = get_model(f'{path}/densenet-bc-100-12-{i}/model_best.pt')
        models_de.append(model)

    torch.save([m.state_dict() for m in models_de], f'checkpoints/{args.dataset.upper()}_de_densenet.pt')


############################
## SANITY CHECKS          ##
############################

path = '/mnt/Data/Datasets'
# path = '/home/ubuntu/Datasets'

# Data
print('==> Preparing dataset %s' % args.dataset)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
if args.dataset == 'cifar10':
    dataloader = datasets.CIFAR10
    num_classes = 10
elif args.dataset == 'svhn':
    dataloader = datasets.SVHN
    num_classes = 10
else:
    dataloader = datasets.CIFAR100
    num_classes = 100

if args.dataset == 'svhn':
    trainset = dataloader(root=path, split='train', download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = dataloader(root=path, split='test', download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
else:
    trainset = dataloader(root=path, train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = dataloader(root=path, train=False, download=False, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)


@torch.no_grad()
def predict(testloader, model):
    outs, targets = [], []

    # switch to evaluate mode
    model.eval()

    for batch_idx, (x, y) in enumerate(testloader):
        x = x.cuda()
        outputs = torch.softmax(model(x), -1)
        outs.append(outputs.cpu())
        targets.append(y)

    outs = torch.cat(outs, dim=0).numpy()
    targets = torch.cat(targets).numpy()

    return outs, targets


if not args.deepens:
    pys, targets = predict(testloader, model)
    test_acc = np.mean(pys.argmax(-1) == targets) * 100
    print('Test Acc:  %.2f' % (test_acc))
else:
    py = 0
    K = len(models_de)

    for k in range(K):
        py_, targets = predict(testloader, models_de[k])
        py += 1/K * py_

    test_acc = np.mean(np.argmax(py, 1) == targets)*100
    print('Test Acc:  %.2f' % (test_acc))
