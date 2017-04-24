#!/usr/bin/python
''' Uses a model with saved weights to classify a set of test images. '''

import argparse
import random

import numpy as np
import tensorflow as tf

from google_net import GoogleNet
from image_producer import ImageProducer

def load_data(csv_file):
    ''' Loads training data metadata from the specified file.

    Note that this function or the file may need to be changed to match the details of your
    metadata/this function.

    Args:
        csv_file: location to file containing training metadata.

    Returns:
        photoids: a list of id's that identify the test images.
        files: a list of image filesnames.
    '''
    photoids = []
    files = []
    with open(csv_file, 'r') as f:
        for line in f:
            # Modify the loop to match your metadata.
            data = line.strip().split('\t')
            photoid.append(data[0])
            fname = data[1]
            files.append(fname)
    return photoids, files

def write_result(f_out, photo_id, photo_name, pred):
    ''' Writes the results to the file f_out '''
    confidence = np.max(pred)
    grid = np.argmax(pred)
    f_out.write('\t'.join([photo_id, photo_name, str(confidence), str(grid)]) + "\n")

def run_classify(metadata_path, num_grids, save_name, saved_model, input_dim=224, num_channels=3,
                 basedir="", output_layer="prob"):
    ''' Loads a trained model and classifies the images specified in metadata_path.

    Args:
        metadata_path: File that contains training metadata (filenames and labels). See load_data.
        num_grids: Number of grids that the world map is divided into.
        save_name: File to save results to.
        saved_model: Location of .npy file containing weights of the trained model.
        input_dim: Input dimension of the network. Currently assumes square dimensions.
        num_channels: Number of channels in the image.
        base_dir: Directory where the training images are located.
        output_layer: Layer from which to take predictions from.
    '''
    # Use None as first dimension for variable sized inputs.
    images = tf.placeholder(tf.float32, [None, input_dim, input_dim, num_channels])
    net = GoogleNet_Places205({'data': images}, num_grids, trainable=[])
    pred = net.layers[output_layer]

    ''' DATA INPUT '''
    # Load Data
    shape = (input_dim, input_dim, num_channels)
    photoids, image_paths = load_data(metadata_path, fname_func)

    # Edit the image producer to fit your pre-processing pipeline.
    image_loader = ImageProducer(image_paths,
                                 batch_size=200,
                                 channels=num_channels,
                                 scale=256,
                                 crop=input_dim,
                                 isotropic=False,
                                 expects_bgr=True,
                                 basedir=basedir,
                                 training=False)

    print("Using model: {}".format(saved_model))

    f_out = open(save_name, 'w')
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        # Shouldn't be ignoring missing because last layer changes
        net.load(saved_model, sess)

        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_loader.start(sess, coordinator)

        try:
            while not coordinator.should_stop():
                idx, input_images = image_loader.get(sess)
                feed = {images: input_images}
                np_pred = sess.run([pred], feed_dict=feed)[0]

                for i in range(len(idx)):
                    write_result(f_out, photoids[idx[i]], image_paths[idx[i]], np_pred[i])
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)
    f_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help='Path to test metadata.')
    parser.add_argument('num_grids', help='Number of output classes.', type=int)
    parser.add_argument('save_name', help='Name to save results as.')
    parser.add_argument('saved_model', help='Saved model weights')
    parser.add_argument('--input-dim', help='Network input size.', type=int, default=224)
    parser.add_argument('--num-channels', help='Number of channels in image.',
                        type=int, default=3)
    parser.add_argument('--base-dir', help="Image location.", default="")
    parser.add_argument('--output-layer', help="Name of prediction layer.", default="pred")
    args = parser.parse_args()

    run_classify(metadata_path=args.data, num_grids=args.num_grids, save_name=args.save_name,
                 saved_model=args.saved_model, input_dim=args.input_dim, num_channels=args.num_channels,
                 basedir=args.base_dir, output_layer=args.output_layer)



