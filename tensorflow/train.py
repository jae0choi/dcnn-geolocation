#!/usr/bin/python
''' Script for training a Tensorflow model for Geolocation. '''
import argparse
import os
import random
import sys

import numpy as np
import tensorflow as tf

##############################
### IMPORT YOUR MODEL HERE ###
##############################
from google_net import GoogleNet
from image_producer import ImageProducer

def one_hot(labels, num_classes):
    return np.eye(num_classes)[labels]

def load_data(csv_file):
    ''' Loads training data metadata from the specified file.

    Note that this function or the file may need to be changed to match the details of your
    metadata/this function.

    Args:
        csv_file: location to file containing training metadata.

    Returns:
        files: a list of image filesnames.
        labels: a list of the corresponding labels for the images.
    '''
    files = []
    labels = []
    with open(csv_file, 'r') as f:
        for line in f:
            # Adjust the lines below to match your metadata.
            data = line.strip().split('\t')
            filename = data[1]
            files.append(filename)
            labels.append(int(data[5]))
    return files, labels

def run_train(train_data, num_grids, save_name, input_dim=224, num_channels=3, batch_size=200,
              base_dir='', saved_model=None, num_epochs=3, epoch_start=0, log_placement=False):
    ''' Loads a pre-defined model and starts training on the layers specified below.

    TODO: Implement support for tf.train.Saver as it would be preferable to .npy files.
    TODO: Add Tensorboard visualization.

    Args:
        train_data: File that contains training metadata (filenames and labels). See load_data.
        num_grids: Number of grids that the world map is divided into.
        save_name: Name to save checkpoints as.
        input_dim: Input dimension of the network. Currently assumes square dimensions.
        num_channels: Number of channels in the image.
        batch_size: The batch size to be used for training.
        base_dir: Directory where the training images are located.
        saved_model: Location of .npy file containing pre-trained model weights.
        num_epochs: Number of epochs to train for.
        epoch_start: Which epoch to start from; useful for restarting training.
        log_placement: Tell tensorflow to log device placement.
    '''

    # Use None as first dimension for variable length batch sizes
    images = tf.placeholder(tf.float32, [None, input_dim, input_dim, num_channels])
    labels = tf.placeholder(tf.float32, [None, num_grids])

    # SPECIFY TRAINABLE VARIABLES HERE
    # If nothing is passed into trainable then all the layers will be trained.
    trainable = [
        'loss3_classifier', 'inception_5b_1x1', 'inception_5b_3x3', 'inception_5b_5x5',
        'inception_5b_pool_proj', 'inception_5b_3x3_reduce', 'inception_5b_5x5_reduce'
    ]
    inputs = {'data': images}
    net = GoogleNet_Places365(inputs, num_grids, trainable=trainable)

    # Get trainable layers for regularization
    trainable_vars = net.get_trainable_variables()

    # Specify the output layer here
    pred = net.layers['prob']

    # Specify the loss
    l2_reg = 1e-2
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output, labels), 0)  + \
        l2_reg * (tf.add_n([tf.nn.l2_loss(weights) for weights in trainable_vars]))

    # CHOOSE YOUR OPTIMIZER HERE
    learning_rate = 0.045
    opt = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    train_op = opt.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

    # DATA LOADING HERE
    # Specify the input dimensions of your conv net here.
    train_images, train_labels = load_data(train_data)

    # Edit the image producer to fit your pre-processing pipeline.
    image_loader = ImageProducer(train_images,
                                 batch_size=batch_size,
                                 channels=num_channels,
                                 scale=256,
                                 crop=input_dim,
                                 isotropic=False,
                                 expects_bgr=True,
                                 labels=train_labels,
                                 basedir=base_dir)

    with tf.Session(config=tf.ConfigProto(log_device_placement=log_placement)) as sess:
        sess.run(tf.initialize_all_variables())

        if saved_model:
            # Ignore missing because last layer changes
            net.load(saved_model, sess, ignore_missing=True)

        print tf.trainable_variables()
        print 'Starting training with weights {}, batch_size: {}, learning_rate: {}'.format(saved_model, batch_size, learning_rate)
        i = 0
        for epoch in range(epoch_start, epoch_start + num_epochs):
            print('Epoch: {0}'.format(epoch))

            # Resets the image loader by reenqueueing all training images.
            if epoch != 0:
                image_loader.setup(batch_size, image_loader.num_threads)

            # Start the image processing workers
            coordinator = tf.train.Coordinator()
            threads = image_loader.start(sess, coordinator)
            try:
                while not coordinator.should_stop():
                    input_labels, input_images = image_loader.get(sess)
                    input_labels = one_hot(input_labels, num_grids)
                    feed = {images: input_images, labels: input_labels}
                    np_loss, np_pred, acc, _  = sess.run([loss, pred, accuracy, train_op], feed_dict=feed)

                    if i % 100 == 0:
                        print(('Iteration:', i, 'Batch loss:', np_loss, 'Batch acc:', acc))
                        sys.stdout.flush() # Flush to stdout to get immediate updates

                    # CHANGE SAVING BEHAVIOR HERE ###
                    # This defines how often the model is saved
                    if i % 1000 == 0:
                        net.save(save_name + '_{0}.npy'.format(epoch), sess)
                    i += 1
            except tf.errors.OutOfRangeError:
                print('Reached epoch: {0}.'.format(epoch))
            finally:
                # When done, ask the threads to stop.
                coordinator.request_stop()

            coordinator.join(threads)
            print('Saving epoch weights..')
            net.save(save_name + '_{0}.npy'.format(epoch), sess)
            print('Epoch over')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help='Path to training data.')
    parser.add_argument('num_grids', help='Number of output classes.', type=int)
    parser.add_argument('save_name', help='Name to save model as.')
    parser.add_argument('--input-dim', help='Network input size.', type=int, default=224)
    parser.add_argument('--num-channels', help='Number of channels in image.',
                        type=int, default=3)
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=200)
    parser.add_argument('--base-dir', help='Image location.', default='')
    parser.add_argument('--saved-model', help='Saved model weights')
    parser.add_argument('--num-epochs', help='Number of epochs.', type=int, default=3)
    parser.add_argument('--epoch-start', help='Epoch to start at. This influences saving behavior',
                        type=int, default=0)
    parser.add_argument('--log-placement', help='Tell tensorflow to log device placement',
                        action='store_true')
    args = parser.parse_args()

    run_train(train_data=args.train_data, num_grids=args.num_grids, save_name=args.save_name,
              input_dim=args.input_dim, num_channels=args.num_channels, batch_size=args.batch_size,
              base_dir=args.base_dir, saved_model=args.saved_model, num_epochs=args.num_epochs,
              epoch_start=args.epoch_start, log_placement=args.log_placement)
