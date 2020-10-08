#!/bin/bash

python cifar.py -a densenet --dataset cifar10 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12

python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12

python cifar.py -a densenet --dataset svhn --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/svhn/densenet-bc-100-12
