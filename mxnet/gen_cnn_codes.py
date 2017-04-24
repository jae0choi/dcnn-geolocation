#!/usr/bin/python
'''
Uses MXNet to fine tune a pretrained model with distribued training capability.
'''
import argparse
import logging
import numpy as np
import os
import sys
import time

import mxnet as mx

import custom_symbols
from skipping_iter import SkippingIter

def get_fine_tune_model(sym, arg_params, args):
    """ Appends a new fully connected layer to layer_name and uses that for prediction.

    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    layer_name = args.layer_name
    all_layers = sym.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.BlockGrad(data=net, name='code_output')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

def get_iterator(itername, args, kv):
    """ Creates and returns DataIters using train_data and val_data based on the args.
    Will use ImageRecordIter or image_iterator depending on whether use_record_iter is set to
    True.
    """
    batch_size = max(args.num_gpus * args.batch_size, args.batch_size)
    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    dtype = np.float32;
    (rank, nworker) = (kv.rank, kv.num_workers)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    data_iter = mx.io.ImageRecordIter(
        path_imgrec         = itername,
        label_width         = 1,
        mean_r              = rgb_mean[0],
        mean_g              = rgb_mean[1],
        mean_b              = rgb_mean[2],
        data_name           = 'data',
        data_shape          = image_shape,
        pad                 = args.pad_size,
        batch_size          = batch_size,
        preprocess_threads  = args.data_nthreads,
        num_parts           = nworker,
        part_index          = rank)
    return data_iter

def load_model(args):
    assert args.saved_model is not None
    model_prefix = args.saved_model
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def get_save_name(args, itername, rank):
    if args.save_name is None:
        logging.info("Must specify save name")
        sys.exit(1)
    dst_dir = os.path.dirname(args.save_name)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return "%s-%s-%d" % (args.save_name, itername, rank)

def run_cnn_codes(args):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    rank = kv.rank
    # For logging.
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('running train!')
    print("batch size per gpu: %d" % args.batch_size)
    batch_size = max(args.batch_size, args.num_gpus * args.batch_size)
    print("total batch size: %d" % batch_size)

    #save_name = get_save_name(args, rank)
    #f_out = open(save_name, 'w')

    # Load pretrained model.
    sym, arg_params, aux_params = load_model(args)
    logging.info('loaded pretrained models..')
    sym, arg_params = get_fine_tune_model(sym, arg_params, args)

    devs = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else mx.cpu()
    mod = mx.mod.Module(symbol=sym, context=devs)

    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    provide_data = [('data', (batch_size,) + image_shape)]
    mod.bind(data_shapes=provide_data, for_training=False)
    mod.set_params(arg_params, aux_params, allow_missing=True)
    logging.info('initialized modules...')

    filenames = {}
    with open("../../../../yfcc100m/recfiles/mediaeval2016_test.lst", 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            idx = int(line[0])
            fname = line[2][:-4]
            filenames[idx] = fname

    # Save CNN Codes.
    for itername in args.data_iters.split(','):
        save_name = get_save_name(args, itername, rank)
        f_out = open(save_name, 'w')
        logging.info("Using {}".format(itername))
        data_iter = get_iterator(args.base_dir + '/' + itername, args, kv)
        tic = time.time()
        num_in_time = 0
        for pred, i_batch, batch in mod.iter_predict(data_iter):
            if i_batch % 10 == 0:
                logging.info("Batch: %d" % (i_batch))
                if i_batch != 0:
                    print("%f samples/sec" % (float(num_in_time)/(time.time() - tic)))
                    num_in_time = 0
                    tic = time.time()
                    #break
            index = batch.index
            pred = pred[0].asnumpy()
            num = pred.shape[0]
            num_in_time += num
            for i in range(num):
                output = pred[i]
                if len(output.shape) > 0:
                    output = output.reshape(-1)
                idx = index[i]
                output_str = ','.join(map(str, output))
                f_out.write("{}\t{}\n".format(filenames[idx], output_str))
        f_out.close()
        

def add_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('RunArgs', 'Args for running MXNet')
    train.add_argument('--layer-name', help="Layer to extract codes from.", default='flatten_0')
    train.add_argument('--save-name', help='Name to save model as.')
    train.add_argument('--batch-size', help='Batch size.', type=int, default=200)
    train.add_argument('--saved-model', help='Saved model weights')
    train.add_argument('--load-epoch', type=int, default=0)
    train.add_argument('--num-gpus', type=int, default=0)
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    return train

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-iters', type=str, help='the training data')
    data.add_argument('--base-dir', type=str, default="")
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--image-shape', type=str, default='3,224,224',
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    add_data_args(parser)
    args = parser.parse_args()
    run_cnn_codes(args)
