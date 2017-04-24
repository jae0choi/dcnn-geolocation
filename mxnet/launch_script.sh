#!/bin/sh
python ~/mxnet/tools/launch.py -n 2 -H hosts \
  python train_dist.py --data-train /home/ubuntu/mxnet/example/image-classification/data/cifar10_train.rec --num-classes 10 \
    --save-name cifar10/test --batch-size 128 --saved-model models/places365_ --num-epochs 100 \
    --num-gpus 1 --data-val /home/ubuntu/mxnet/example/image-classification/data/cifar10_val.rec \
    --kv-store dist_device_sync --image-shape 3,224,224 --pad-size 98 --val-every 1 