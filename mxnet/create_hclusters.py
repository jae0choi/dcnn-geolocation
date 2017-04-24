import argparse
import logging
logging.basicConfig(level=logging.DEBUG)

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hcluster
import sklearn.cluster as skc

def load_data(csv_file):
    """ Loads training data metadata from the specified file.
    This file should be changed to fit your training metadata """
    imglist = []
    latlng = []
    with open(csv_file, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            filename = data[1]
            lng = float(data[2])
            lat = float(data[3])
            imglist.append(filename)
            latlng.append([lng, lat])
    return np.array(imglist), np.array(latlng)

def get_cluster_centers(clusters, cluster_data):
    num_clusters = np.max(clusters)
    centers = np.zeros((num_clusters, 2))
    counts = np.zeros(num_clusters)
    for i in range(clusters.shape[0]):
        c = clusters[i] - 1
        coords = cluster_data[i]
        centers[c] += coords
        counts[c] += 1
    return centers / counts[:, None]

def save_data(fname, imglist, labels):
    with open(fname, 'w') as f:
        for i in range(len(imglist)):
            f.write("{}\t{}\t{}\n".format(i, labels[i], imglist[i]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', help='Path to training data.')
    parser.add_argument('val_data', help='Path to val data.')
    parser.add_argument('test_data', help='Path to test data.')
    parser.add_argument('prefix')
    parser.add_argument('--thresh', type=float, default=1.5)
    parser.add_argument('--thin', type=float, default=1.0)
    args = parser.parse_args()

    # Load and shuffle data
    train_imgs, train_points = load_data(args.train_data)
    num_train = len(train_imgs)
    num_thinned = int(args.thin * num_train)
    idx = np.arange(num_train)
    np.random.shuffle(idx)
    train_imgs = train_imgs[idx]
    train_points = train_points[idx]
    cluster_data = train_points[:num_thinned]

    logging.info("Finished loading training data.")

    # clustering
    thresh = args.thresh
    clusters = hcluster.fclusterdata(cluster_data, thresh, criterion="distance")
    logging.info("Finished clustering.")
    num_clusters = np.max(clusters)
    centers = get_cluster_centers(clusters, cluster_data)
    logging.info("Num clusters: {}".format(num_clusters))

    kmeans = skc.KMeans(num_clusters)
    kmeans.fit(centers)
    train_labels = kmeans.predict(train_points)
    centers = kmeans.cluster_centers_

    # Save clusters.
    with open(prefix + ".clusters", 'w') as f:
        for i in range(centers.shape[0]):
            latlng = centers[i]
            lat = latlng[1]
            lng = latlng[0]
            f.write("{}\t{}\t{}\n".format(i, lat, lng))
    # Save data.
    prefix += "-{}-{}".format(args.thresh, args.thin)
    save_data(prefix + '-train.lst', train_imgs, train_labels)
    logging.info("Finished saving training data.")
    
    val_imgs, val_points = load_data(args.val_data)
    val_labels = kmeans.predict(val_points)
    save_data(prefix + '-val.lst', val_imgs, val_labels)
    test_imgs, test_points = load_data(args.test_data)
    test_labels = kmeans.predict(test_points)
    save_data(prefix + '-test.lst', test_imgs, test_labels)
