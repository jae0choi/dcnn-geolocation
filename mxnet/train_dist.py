#!/usr/bin/python
'''
Uses MXNet to fine tune a pretrained model with distribued training capability.
'''
import argparse
import logging
import numpy as np
import os
import time

os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
import mxnet as mx

from skipping_iter import SkippingIter

def get_fine_tune_model(sym, arg_params, args):
    """ Appends a new fully connected layer to layer_name and uses that for prediction.

    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = sym.get_internals()
    layer_name = args.layer_name
    if args.custom_loss is not None:
        layer_name = 'fc1'
    net = all_layers[layer_name+'_output']
    if args.custom_loss is not None:
        import custom_symbols
        # Name is softmax because of the iterator label name, will change later
        logging.info("Using custom loss: %s" % args.custom_loss)
        net = mx.symbol.Custom(data=net, name='softmax', op_type=args.custom_loss)
    else:
        net = mx.symbol.FullyConnected(data=net, num_hidden=args.num_classes, name='fc1')
        net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

def get_iterators(args, kv):
    """ Creates and returns DataIters using train_data and val_data based on the args.
    Will use ImageRecordIter or image_iterator depending on whether use_record_iter is set to
    True.
    """
    batch_size = max(args.num_gpus * args.batch_size, args.batch_size)

    image_shape = tuple([int(l) for l in args.image_shape.split(',')])
    dtype = np.float32;
    (rank, nworker) = (kv.rank, kv.num_workers)
    rgb_mean = [float(i) for i in args.rgb_mean.split(',')]
    do_nothing_std = np.array([1., 1., 1.])
    if args.skip_iter:
        logging.info("Using skipping iter.")
        train = SkippingIter(
            batch_size          = batch_size,
            data_shape          = image_shape,
            label_width         = 1,
            path_imgrec         = args.data_train + ".rec",
            path_imglist        = args.train_labels,
            path_imgidx         = args.data_train + ".idx",
            shuffle             = False,
            part_index          = rank,
            num_parts           = nworker,
            data_name           = 'data',
            label_name          = 'softmax_label',
            mean                = rgb_mean,
            std                 = do_nothing_std)
        if args.data_val is None:
            return (train, None)
        val = SkippingIter(
            batch_size          = batch_size,
            data_shape          = image_shape,
            label_width         = 1,
            path_imgrec         = args.data_val + ".rec",
            path_imglist        = args.val_labels,
            path_imgidx         = args.data_val + ".idx",
            shuffle             = False,
            part_index          = rank,
            num_parts           = nworker,
            data_name           = 'data',
            label_name          = 'softmax_label',
            mean                = rgb_mean,
            std                 = do_nothing_std)
        return (train, val)
    else:
        train = mx.io.ImageRecordIter(
            path_imglist        = args.train_labels,
            path_imgrec         = args.data_train,
            label_width         = 1,
            mean_r              = rgb_mean[0],
            mean_g              = rgb_mean[1],
            mean_b              = rgb_mean[2],
            data_name           = 'data',
            label_name          = 'softmax_label',
            data_shape          = image_shape,
            pad                 = args.pad_size,
            batch_size          = batch_size,
            preprocess_threads  = args.data_nthreads,
            shuffle             = True,
            num_parts           = nworker,
            part_index          = rank)
        if args.data_val is None:
            return (train, None)
        val = mx.io.ImageRecordIter(
            path_imglist        = args.val_labels,
            path_imgrec         = args.data_val,
            label_width         = 1,
            mean_r              = rgb_mean[0],
            mean_g              = rgb_mean[1],
            mean_b              = rgb_mean[2],
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = batch_size,
            data_shape          = image_shape,
            pad                 = args.pad_size,
            preprocess_threads  = args.data_nthreads,
            num_parts           = nworker,
            part_index          = rank)
    return (train, val)

def intermediate_checkpoint_callback(mod, args, save_prefix):
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
        if param.nbatch % args.save_every == 0:
            logging.info("Saving checkpoint..")
            mod.save_checkpoint(save_prefix, param.epoch + 1, args.not_dist)
    return _callback

def evaluation_callback(mod, args, val_iter, f_out=None):
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
        epoch, nbatch = param.epoch, param.nbatch
        if nbatch % args.eval_every == 0:
            # Can I use iter_predict here? It might be lazy evaluation..
            for i in range(5):
                batch = val_iter.next()
                mod.forward(batch, is_train=False)
                prob = mod.get_outputs()[0].asnumpy()
                pred = np.argmax(prob, 1)
                acc = float(np.sum(pred == batch.label[0].asnumpy())) / pred.shape[0]
                message = 'Epoch[%d] Batch[%d] Validation-Acc=%f' % (epoch, nbatch, acc)
                logging.info(message)
                if f_out:
                    f_out.write(message + '\n')
    return _callback

def load_model(args, rank):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.saved_model is not None
    model_prefix = args.saved_model
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def get_save_name(args, rank):
    if args.save_name is None:
        return None
    dst_dir = os.path.dirname(args.save_name)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return args.save_name if rank == 0 else "%s-%d" % (args.save_name, rank)

def run_train(args):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    rank = kv.rank
    # For logging.
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if args.logfile != "":
        logging.basicConfig(filename="{}_{}".format(args.logfile, rank) ,level=logging.DEBUG, format=head)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('running train!')
    f_out = open('{}_{}'.format(args.val_save, rank), 'a')

    print("batch size per gpu: %d" % args.batch_size)
    batch_size = max(args.batch_size, args.num_gpus * args.batch_size)
    print("total batch size: %d" % batch_size)

    # Create image iterator.
    train_iter, val_iter = get_iterators(args, kv)
    logging.info('created iterators..')
    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train_iter):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()
        return

    # Load pretrained model.
    # sym, arg_params, aux_params = mx.model.load_checkpoint(args.saved_model, args.epoch_start)
    sym, arg_params, aux_params = load_model(args, rank)
    logging.info('loaded pretrained models..')

    # Create model for fine tuning.
    if args.epoch_start == 0:
        sym, arg_params = get_fine_tune_model(sym, arg_params, args)

    fixed_params = None
    if args.fine_tune is not None:
        layers = args.fine_tune.split(',')
        fixed_params = [k for k in arg_params]
        for l in layers:
            fixed_params = [p for p in fixed_params if l not in p]

    devs = [mx.gpu(i) for i in range(args.num_gpus)] if args.num_gpus > 0 else mx.cpu()
    mod = mx.mod.Module(symbol=sym, context=devs, fixed_param_names=fixed_params)
    mod.bind(data_shapes=train_iter.provide_data, label_shapes=train_iter.provide_label)
    mod.init_params(initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2))
    mod.set_params(arg_params, aux_params, allow_missing=True)
    if args.load_epoch > 0:
        logging.info('restoring optimizer state..')
        # Need to set mod._preload_opt_states because optimizer is not initialized yet and
        # mod.load_optimizer_states will throw an error.
#        mod._preload_opt_states = '%s-%04d.states'%(args.saved_model, args.load_epoch)
    metric = mx.metric.Accuracy()
    logging.info('initialized modules...')

    # Create callbacks.
    eval_callback = evaluation_callback(mod, args, val_iter, f_out)
    save_prefix = get_save_name(args, rank)
    checkpoint = mx.callback.module_checkpoint(mod, save_prefix, save_optimizer_states=args.not_dist)
    intermediate_checkpoint = intermediate_checkpoint_callback(mod, args, save_prefix)

    batch_end_callbacks = [mx.callback.Speedometer(batch_size, args.disp_batches),
                           intermediate_checkpoint, eval_callback]
    epoch_end_callbacks = [checkpoint]

    lr_scheduler = mx.lr_scheduler.FactorScheduler(20000, 0.7)
    # Train!
    logging.info('Calling fit...')
    mod.fit(train_iter,
        num_epoch=args.num_epochs,
        begin_epoch=args.epoch_start,
        batch_end_callback=batch_end_callbacks,
        epoch_end_callback=epoch_end_callbacks,
        kvstore=kv,
        optimizer='sgd',
        optimizer_params={'learning_rate':args.lr, 'wd':args.reg, 'momentum': 0.9,
            'lr_scheduler': lr_scheduler},
        eval_metric='acc')

    acc = mod.score(val_iter, metric)[0][1]
    print("Final validation accuracy %.3f" % acc)
    f_out.close()

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--save-name', help='Name to save model as.')
    train.add_argument('--batch-size', help='Batch size.', type=int, default=200)
    train.add_argument('--saved-model', help='Saved model weights')
    train.add_argument('--val-save', help='File to save val name to.')
    #train.add_argument('--num-epochs', help='Number of epochs.', type=int, default=3)
    train.add_argument('--epoch-start', type=int, default=0)
    train.add_argument('--load-epoch', type=int, default=0)
    train.add_argument('--num-gpus', type=int, default=0)
    train.add_argument('--save-every', help="How often to save checkpoints.", type=int,
                        default=1000)
    train.add_argument('--eval-every', help="How often to test on Validation set", type=int,
                       default=500)
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.01,
                       help='initial learning rate')
    train.add_argument('--reg', type=float, default=0.0001,
                       help='L2 Regularization Constant')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    train.add_argument('--logfile', type=str, default="",
                       help='Filename for logging.')
    train.add_argument('--custom-loss', type=str, help="Name of custom op defined in custom_symbols.py.")
    train.add_argument('--fine-tune', type=str, help="Comma-separated list of layer names to train.")
    train.add_argument('--layer-name', type=str, default="flatten_0")
    train.add_argument('--not-dist', action='store_true')
    return train

def add_data_args(parser):
    data = parser.add_argument_group('Data', 'the input images')
    data.add_argument('--data-train', type=str, help='the training data')
    data.add_argument('--data-val', type=str, help='the validation data')
    data.add_argument('--train-labels', default='')
    data.add_argument('--val-labels', default='')
    data.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                      help='a tuple of size 3 for the mean rgb')
    data.add_argument('--image-shape', type=str, default='3,224,224',
                      help='the image shape feed into the network, e.g. (3,224,224)')
    data.add_argument('--num-classes', type=int, help='the number of classes')
    data.add_argument('--data-nthreads', type=int, default=4,
                      help='number of threads for data decoding')
    data.add_argument('--pad-size', type=int, default=0,
                      help='padding the input image')
    data.add_argument('--skip-iter', action="store_true")
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_fit_args(parser)
    add_data_args(parser)
    args = parser.parse_args()
    run_train(args)
