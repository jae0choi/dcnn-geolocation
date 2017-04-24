""" Benchmarking different forms of loading images """

from __future__ import print_function
import os
import time
# set the number of threads you want to use before importing mxnet
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '32'
import mxnet as mx
import numpy as np
import cv2
from image_iterator import ImageIter

import logging
logging.basicConfig(level=logging.DEBUG)

def load_data(csv_file):
    """ Loads training data metadata from the specified file.
    This file should be changed to fit your training metadata """
    imglist = []
    with open(csv_file, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            filename = data[4]
            filename = filename[0:3] + '/' + filename[3:6] + '/' + filename + '.jpg'
            imglist.append([float(data[-1]), filename])
    return imglist

imglist = load_data("data/toy.train")
imgs = list(map(lambda x: x[1], imglist))
base_dir = "../../../../yfcc100m/images"
do_nothing_std = np.array([1.0, 1.0, 1.0])
mean = np.array([123.68, 116.28, 103.53])

# opencv
N = 1000
tic = time.time()
for i in range(N):
    img = cv2.imread(os.path.join(base_dir, imgs[i]), flags=1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(N/(time.time()-tic), 'images decoded per second with opencv')

# mx.image
tic = time.time()
for i in range(N):
    img = mx.image.imdecode(open(os.path.join(base_dir, imgs[i])).read())
mx.nd.waitall()
print(N/(time.time()-tic), 'images decoded per second with mx.image')

def resize_square(size, interp=2):
    def aug(src):
        return [mx.image.imresize(src, size, size, interp=interp)]
    return aug

auglist = []
auglist.append(resize_square(256))
crop_size = (224, 224)
auglist.append(mx.image.CenterCropAug(crop_size))
auglist.append(mx.image.CastAug())
auglist.append(mx.image.ColorNormalizeAug(mean, do_nothing_std))
train_iter = ImageIter(batch_size=200,
                       data_shape=(3, 224, 224),
                       label_width=1,
                       path_root=base_dir,
                       aug_list=auglist,
                       imglist=imglist)

train_iter2 = ImageIter(batch_size=200,
                       data_shape=(3, 224, 224),
                       label_width=1,
                       path_root=base_dir,
                       aug_list=auglist,
                       imglist=imglist)
prefetch_iter = mx.io.PrefetchingIter([train_iter2])

# mx.image
tic = time.time()
for _ in range(5): train_iter.next()
print(N/(time.time()-tic), 'images decoded per second with ImageIter')

# mx.image
tic = time.time()
for _ in range(5): prefetch_iter.next()
print(N/(time.time()-tic), 'images decoded per second with prefetch ImageIter')
