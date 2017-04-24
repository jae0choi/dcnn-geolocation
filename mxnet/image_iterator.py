import logging
import os
import sys
import random

# set the number of threads you want to use before importing mxnet
#os.environ['MXNET_CPU_WORKER_NTHREADS'] = '32'
import mxnet as mx
import numpy as np

numeric_types = (float, int, np.float32, np.int32)

import time

#def one_hot(label, num_classes):
    # mx.nd.one_hot not working?
#    retval = mx.nd.zeros(num_classes)
#    retval[label] = 1.0
#    return retval

class ImageIter(mx.io.DataIter):
    """Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.
    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.
    To load from raw image files, specify path_imglist and path_root.
    Parameters
    ----------
    batch_size : int
        Number of examples per batch
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : int
        dimension of label
    imglist: list
        a list of image with the label(s)
        each item is a list [imagelabel: float or list of float, imgpath]
    path_root : str
        Root folder of image files
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    kwargs : ...
        More arguments for creating augumenter. See mx.image.CreateAugmenter
    """
    def __init__(self, batch_size, data_shape, label_width=1, path_root=None,
                 shuffle=False, aug_list=None, imglist=None, num_classes=1, **kwargs):
        super(ImageIter, self).__init__()
        print('loading image list...')
        sys.stdout.flush()
        result = {}
        imgkeys = []
        index = 1
        for img in imglist:
            key = str(index)
            index += 1
            if isinstance(img[0], numeric_types):
                label = [img[0]]
            else:
                label = mx.nd.array(img[0])
            result[key] = (label, img[1])
            imgkeys.append(str(key))
        self.imglist = result
        self.path_root = path_root
        print('Done processing imglist')
        sys.stdout.flush()

        assert len(data_shape) == 3 and data_shape[0] == 3
        self.provide_data = [('data', (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [('softmax_label', (batch_size, label_width))]
        else:
            self.provide_label = [('softmax_label', (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        self.seq = imgkeys

        self.print_batch = True
        if aug_list is None:
            self.auglist = mx.image.CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        self.cur = 0

    def next_sample(self):
        """helper function for reading in next sample"""
        #logging.info("sample..")
        if self.cur >= len(self.seq):
            raise StopIteration
        idx = self.seq[self.cur]
        self.cur += 1
        label, fname = self.imglist[idx]
        imgname = os.path.join(self.path_root, fname)
        with open(imgname) as fin:
            img = fin.read()
 #       if self.label_width > 1 and len(label) == 1:
 #           grid = int(label[0])
 #           #print(grid)
 #           label = one_hot(grid, self.label_width)
 #           #print(label.asnumpy())
 #           #print(label.asnumpy()[grid])
        if self.label_width == 1: label = label[0]
        return label, img

    def next(self):
        if self.print_batch: logging.info("Getting first batch..")
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = mx.nd.empty((batch_size, c, h, w))
        batch_label = mx.nd.empty(self.provide_label[0][1])
        i = 0
        j = 0
        ind = True
        try:
            while i < batch_size:
                if self.print_batch and j % 100 == 0:
                    logging.info("Tried %d, loaded %d.." % (j, i))
                j += 1
                try:
                    # Wrap line below in try/except to ignore corrupted images.
                    label, s = self.next_sample()
                except Exception as e:
                    if isinstance(e, StopIteration): raise StopIteration
                    #logging.debug("{} couldn't find image image, skipping..".format(e))
                    continue
                data = [mx.image.imdecode(s)]
                if len(data[0].shape) == 0:
                    logging.debug('Invalid image, skipping.')
                    continue
                for aug in self.auglist:
                    data = [ret for src in data for ret in aug(src)]
                for d in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = mx.nd.transpose(d, axes=(2, 0, 1))
                    batch_label[i][:] = label
                    i += 1
            mx.nd.waitall()
        except StopIteration:
            if not i:
                raise StopIteration
        if self.print_batch:
            self.print_batch = False
            print(batch_label.asnumpy())
            logging.info("Finished loading batch")
            logging.info("Attemped to load %d images" % j)
        return mx.io.DataBatch([batch_data], [batch_label], batch_size-i)
