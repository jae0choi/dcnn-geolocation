#!/usr/bin/python
'''
Analyzes results
'''
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from math import radians, sin, cos, sqrt, asin

LEVELS= [0.1, 1, 10, 100, 1000, 10000, "10000+"]

def print_stats(distances):
    print(("Shortest dist:", distances[0], "Longest dist:", distances[-1]))
    percs = []
    for i in range(len(LEVELS)):
        if i == 0:
            num_level = np.sum(distances <= LEVELS[i])
        elif i == (len(LEVELS) - 1):
            num_level = np.sum(distances > LEVELS[i-1])
        else:
            in_range = (distances > LEVELS[i-1]) & (distances <= LEVELS[i])
            num_level = np.sum(in_range)

        perc = (num_level / float(len(distances))) * 100
        percs.append(perc)
        print "{}".format(perc)
    print "TOTAL: {}".format(np.sum(np.array(percs)))

def plot_distances(distances, save, top_distances=None, top_k=0):
    distances = np.array(sorted(distances))
    print("TOP 1:")
    print_stats(distances)
    if top_distances is not None:
        top_distances = np.array(sorted(top_distances))
        print("TOP 5:")
        print_stats(top_distances)
    return 
    x = np.arange(len(distances)).astype(float) / len(distances) * 100
    y_ticks = np.array(LEVELS[:-1]).astype(float)
    fig, ax = plt.subplots()
    plt.semilogy(x, distances, label="top 1")
    if top_distances is not None: plt.semilogy(x, top_distances, label="top %d" % top_k)
    plt.grid(True)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_ticks(y_ticks)
    plt.ylim([.1, LEVELS[:-1][-1]])
    plt.xlabel("Percentage of set")
    plt.ylabel("Geolocation Error (log km)")
    #title = "tiltshift "+ " (centroid)"
    plt.legend()
    #plt.title(title)
    if save:
        plt.savefig(save + ".centroid" + ".png")
    else:
        plt.show()

class Point(object):
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    def distance(self, point):
        R = 6371 # Earth radius in km
        lat1, lng1, lat2, lng2 = map(radians, (self.lat, self.lng, point.lat, point.lng))
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat * 0.5) ** 2 + cos(lat1) * cos(lat2) * (sin(dlng * 0.5) ** 2)
        return 2 * R * asin(sqrt(a))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("lst", help=".lst file for going from index to image.")
    parser.add_argument("pred", help="Predictions with index and top5 predictions.")
    parser.add_argument("grids", help="A list of grids.")
    parser.add_argument("latlng", help="Test data with GPS coords and grid labels.")
    parser.add_argument("--top", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--acc", action="store_true")
    parser.add_argument("--outdoor")
    args = parser.parse_args()

    idx_to_img = dict()
    with open(args.lst, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            idx = int(line[0])
            img = line[-1].split('/')[-1][:-4]
            idx_to_img[idx] = img

    top_k = 0
    predictions = dict()
    with open(args.pred, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            idx = int(data[0])
            img = idx_to_img[idx]
            if args.top:
                preds = [int(data[i]) for i in range(1, len(data), 2)]
                top_k = len(preds)
            else:
                preds = [int(data[1])]
            predictions[img] = preds

    grid_centers = dict()
    with open(args.grids, 'r') as f:
        i = 0
        for line in f:
            data = line.strip().split('\t')
            lat = float(data[3])
            lng = float(data[4])
            grid_centers[i] = Point(lat, lng)
            i += 1
        print i

    latlng_labels = dict()
    grid_labels = dict()
    with open(args.latlng, 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            lat = float(data[3])
            lng = float(data[2])
            img = data[4]
            if args.acc:    
                grid = int(data[-1])
                grid_labels[img] = grid
            latlng_labels[img] = Point(lat, lng)

    include = set(predictions.keys())
    if args.outdoor is not None:
        include = []
        with open(args.outdoor, 'r') as f:
            for line in f:
                img = line.split('\t')[1]
                include.append(img)
        include = set(include)

    f_out = open(args.pred + '.close', 'w')

    skipped = 0
    distances = []
    top_distances = None
    if args.top: top_distances = []
    if args.acc: total, top1_correct, top5_correct = 0.0, 0.0, 0.0
    for img, preds in predictions.iteritems():
        if img not in include:
            skipped += 1
            continue
        grids = [grid_centers[grid] for grid in preds]
        ground_truth = latlng_labels[img]
        d = [g.distance(ground_truth) for g in grids]
        distances.append(d[0])
        #if d[0] < 0.1:
        imname = img[0:3] + '/' + img[3:6] + '/' + img + '.jpg'
        f_out.write("{}\t{}\n".format(imname, d[0]))
        if args.top: top_distances.append(min(d))
        if args.acc:
            total += 1
            top1_correct += int(preds[0] == grid_labels[img])
            top5_correct += int(grid_labels[img] in preds)
    if args.acc:
        acc_str = "Top1 Acc: %f\tTop5 Acc: %f" % (top1_correct/total, top5_correct/total)
        print acc_str
    if args.save:
        save = args.pred
    else:
        save = False

    print("Skipped: {}".format(skipped))
    plot_distances(distances, save, top_distances, top_k)
