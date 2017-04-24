#!/usr/bin/python
'''
Uses MXNet to fine tune a pretrained model.
'''

import argparse
import logging
import numpy as np
import os
import random
import sys # Import sys to flush to stdout for more immediate training updates.

# set the number of threads you want to use before importing mxnet
# Not sure if this is necessary here, but safer to include in both here and image_iterator
os.environ['MXNET_CPU_WORKER_NTHREADS'] = '8'
import mxnet as mx
from mxnet._ndarray_internal import _cvimresize as imresize
from image_iterator import ImageIter

head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def load_data(csv_file):
    """ Loads training data metadata from the specified file.
    This file should be changed to fit your training metadata """
    imglist = []
    with open(csv_file, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            filename = data[4]
            if filename[0] >= 'b' and filename[0] <='f': continue
            filename = filename[0:3] + '/' + filename[3:6] + '/' + filename + '.jpg'
            imglist.append([float(data[-1]), filename])
    print("----------Done Loading")
    sys.stdout.flush()
    return imglist

def get_fine_tune_model(sym, arg_params, num_classes, layer_name='flatten_0'):
    """ Appends a new fully connected layer to layer_name and uses that for prediction.

    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

def ResizeSquareAug(size, interp=2):
    def aug(src):
        return [mx.image.imresize(src, size, size, interp=interp)]
    return aug

def get_iterators(train_data, val_data, shape, batch_size, base_dir,
                  train_labels, val_labels, use_record_iter):
    """ Creates and returns DataIters using train_data and val_data based on the args.
    Will use ImageRecordIter or image_iterator depending on whether use_record_iter is set to
    True.
    """
    do_nothing_std = np.array([1., 1., 1.])
    mean = np.array([123.68, 116.28, 103.53])

    train_iter, val_iter = None, None
    if use_record_iter:
        logging.info("Using Record Iters: {}, {}".format(train_data, val_data))
        train_iter = mx.io.ImageRecordIter(
            path_imglist        = train_labels,
            path_imgrec         = train_data,
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = batch_size,
            data_shape          = shape,
            shuffle             = True,
            rand_mirror         = False,
            mean_r              = mean[0],
            mean_g              = mean[1],
            mean_b              = mean[2])
        if val_data:
            val_iter = mx.io.ImageRecordIter(
                path_imglist        = val_labels,
                path_imgrec         = val_data,
                data_name           = 'data',
                label_name          = 'softmax_label',
                batch_size          = batch_size,
                data_shape          = shape,
                rand_crop           = False,
                rand_mirror         = False,
                mean_r              = mean[0],
                mean_g              = mean[1],
                mean_b              = mean[2])
    else:
        auglist = []
        auglist.append(ResizeSquareAug(256))
        crop_size = (shape[2], shape[1])
        auglist.append(mx.image.CenterCropAug(crop_size))
        auglist.append(mx.image.CastAug())
        auglist.append(mx.image.ColorNormalizeAug(mean, do_nothing_std))

        imglist = load_data(train_data)
        train_iter = ImageIter(batch_size=batch_size,
                               data_shape=shape,
                               label_width=1,
                               path_root=base_dir,
                               shuffle=True,
                               aug_list=auglist,
                               imglist=imglist)
        train_iter = mx.io.PrefetchingIter([train_iter])

        if val_data:
            val_imglist = load_data(val_data)
            val_iter = ImageIter(batch_size=batch_size,
                                 data_shape=shape,
                                 label_width=1,
                                 path_root=base_dir,
                                shuffle=True,
                                 aug_list=auglist,
                                 imglist=val_imglist)
    return train_iter, val_iter

def intermediate_checkpoint_callback(mod, prefix, save_every_x_batches=1000):
    """Callback to save params during an epoch.
    ----------
    save_every_x_batches : int
        How often to save checkpoints in terms of batches.
    Returns
    -------
    callback : function
        The callback function that can be passed as batch_end_callback to fit.
    """
    def _callback(param):
        if param.nbatch % save_every_x_batches == 0:
            logging.info("Saving checkpoint..")
            mod.save_checkpoint(prefix, param.epoch + 1, True)
    return _callback

def evaluation_callback(mod, val_iter, metric, f_out=None, eval_every_x_epochs=10):
    """Callback to save params during an epoch.
    ----------
    save_every_x_batches : int
        How often to save checkpoints in terms of batches.
    Returns
    -------
    callback : function
        The callback function that can be passed as batch_end_callback to fit.
    """
    first = [False] # Set to true if you want to get validation accuracy on the first batch.
    def _callback(epoch, sym=None, arg=None, aux=None):
        #nonlocal first # nonlocal doesn't exist before python3.X..
        if epoch % eval_every_x_epochs == 0 or first[0]:
            first[0] = False
            val_score = mod.score(val_iter, metric)[0][1]
            message = 'Epoch[%d] Validation-%s=%f' % (epoch, metric.name, val_score)
            logging.info(message)
            if f_out:
                f_out.write(message + '\n')
    return _callback

def run_train(train_data, num_grids, save_name, input_dim=224, num_channels=3, batch_size=200,
              base_dir="", saved_model=None, num_epochs=3, epoch_start=0, num_gpus=0,
              val_data=None, train_labels=None, val_labels=None, use_record_iter= False,
              save_every_x_batches=1000):
    logging.info('running train!')
    # For evaluation logging.
    f_out = open('validation_scores', 'a')

    print("batch size per gpu: %d" % batch_size)
    batch_size = num_gpus * batch_size
    print("total batch size: %d" % batch_size)

    # Create image iterator.
    shape = (num_channels, input_dim, input_dim)
    train_iter, val_iter = get_iterators(train_data, val_data, shape, batch_size,
                                         base_dir, train_labels, val_labels, use_record_iter)
    logging.info('created iterators..')

    # Load pretrained model.
    sym, arg_params, aux_params = mx.model.load_checkpoint(saved_model, epoch_start)
    logging.info('loaded pretrained models..')

    # Create model for fine tuning.
    if epoch_start == 0:
        sym, arg_params = get_fine_tune_model(sym, arg_params, num_grids)
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(arg_params, aux_params, allow_missing=True)
    if epoch_start > 0:
        logging.info('restoring optimizer state..')
        # Need to set mod._preload_opt_states because optimizer is not initialized yet and
        # mod.load_optimizer_states will throw an error.
        mod._preload_opt_states = '%s-%04d.states'%(saved_model, epoch_start)
    metric = mx.metric.Accuracy()
    logging.info('initialized modules...')

    # Create callbacks for saving the model.
    metric = mx.metric.Accuracy()
    eval_callback = evaluation_callback(mod, val_iter, metric, f_out, 1)
    checkpoint = mx.callback.module_checkpoint(mod, save_name, save_optimizer_states=True)
    intermediate_checkpoint = intermediate_checkpoint_callback(mod, save_name, save_every_x_batches)

    batch_end_callbacks = [mx.callback.Speedometer(batch_size, 10), intermediate_checkpoint]
    epoch_end_callbacks = [checkpoint, eval_callback]

    logging.info('Calling fit...')

    # Train!
    mod.fit(train_iter,
        num_epoch=num_epochs,
        begin_epoch=epoch_start,
        batch_end_callback=batch_end_callbacks,
        epoch_end_callback=epoch_end_callbacks,
        kvstore='device',
        optimizer='adagrad',
        optimizer_params={'learning_rate':0.045, 'wd':0.00002},
        eval_metric='acc')

    acc = mod.score(val_iter, metric)[0][1]
    print("Final validation accuracy %.3f" % acc)
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help='Path to training data.')
    parser.add_argument('num_grids', help='Number of output classes.', type=int)
    parser.add_argument('save_name', help='Name to save model as.')
    parser.add_argument('--input-dim', help='Network input size.', type=int, default=224)
    parser.add_argument('--num-channels', help='Number of channels in image.',
                        type=int, default=3)
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=200)
    parser.add_argument('--base-dir', help="Image location.", default="")
    parser.add_argument('--saved-model', help='Saved model weights')
    parser.add_argument('--num-epochs', help='Number of epochs.', type=int, default=3)
    parser.add_argument('--epoch-start', type=int, default=0)
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--val-data')
    parser.add_argument('--train-labels', default='')
    parser.add_argument('--val-labels', default='')
    parser.add_argument('--record-iter', action='store_true')
    parser.add_argument('--save-every', help="How often to save checkpoints.", type=int,
                        default=1000)
    args = parser.parse_args()

    run_train(train_data=args.train_data,
              num_grids=args.num_grids,
              save_name=args.save_name,
              input_dim=args.input_dim,
              num_channels=args.num_channels,
              batch_size=args.batch_size,
              base_dir=args.base_dir,
              saved_model=args.saved_model,
              num_epochs=args.num_epochs,
              epoch_start=args.epoch_start,
              num_gpus=args.num_gpus,
              val_data=args.val_data,
              train_labels=args.train_labels,
              val_labels=args.val_labels,
              use_record_iter=args.record_iter,
              save_every_x_batches=args.save_every)
