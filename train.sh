#!/bin/bash


# LULA
python cifar.py -a densenet --dataset cifar10 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/densenet-bc-100-12

python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/densenet-bc-100-12

python cifar.py -a densenet --dataset svhn --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/svhn/densenet-bc-100-12


# MCD
python cifar.py -a densenet --dataset cifar10 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --dropout 0.2 --checkpoint checkpoints/cifar10/dropout/densenet-bc-100-12

python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --dropout 0.2 --checkpoint checkpoints/cifar100/dropout/densenet-bc-100-12

python cifar.py -a densenet --dataset svhn --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --dropout 0.2 --checkpoint checkpoints/svhn/dropout/densenet-bc-100-12


# DE  (Each with random random-seed)
for i in {1..4}
do
    python cifar.py -a densenet --dataset cifar10 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar10/deepens/densenet-bc-100-12-$i
done

for i in {1..4}
do
    python cifar.py -a densenet --dataset cifar100 --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/cifar100/deepens/densenet-bc-100-12-$i
done

for i in {1..4}
do
    python cifar.py -a densenet --dataset svhn --depth 100 --growthRate 12 --train-batch 128 --epochs 300 --schedule 150 225 --wd 1e-4 --gamma 0.1 --checkpoint checkpoints/svhn/deepens/densenet-bc-100-12-$i
done
