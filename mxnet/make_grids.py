#!/usr/bin/python
''' Takes lat and lon range and partitions MediaEval data into grids
using Google's S2 geo library (https://github.com/micolous/s2-geometry-library).
'''
import argparse

import numpy as np
import s2

def open_dataset(filename):
    image_data = []
    with open(filename, 'r') as f:
        for line in f:
            # Edit this loop to match your metadata.
            data = line.split('\t')
            data[-1] = data[-1].strip() # Remove new line character
            image_data.append(Image(data))
    return image_data

def make_grids(image_data, t1, t2, val_data=None):
    ''' Makes grids as specified in PlaNet paper. See process.

    Args:
        image_data: File containing the metadata of your training images.
        t1: the t1 parameter specified above.
        t2: the t2 parameter specified above.
        val_data: If specified, this will bin your validation at the same time as well.
    '''
    # Get the first level 0 node
    begin = s2.S2CellId_Begin(0)
    # Get the 5 other top level nodes
    roots = [begin]
    for i in range(5):
        roots.append(roots[-1].next())

    # Divide cells by root
    root_data = bin_data(roots, image_data)
    root_val_data = bin_data(roots, val_data) if val_data else None
    # Run recurisve algorithm on each root
    return build_grids(roots, root_data, t1, t2, root_val_data)


# Separate function for recursive make grid calls because of different
# syntax for retrieving children vs. tree roots.
def _make_grids(node, image_data, t1, t2, val_data):
    if len(image_data) < t1 or node.is_leaf():
        if len(image_data) < t2: return {}
        return {node : image_data}, {node : val_data}
    children = [node.child(i) for i in range(4)]
    children_data = bin_data(children, image_data)
    # Validation data for labeling
    children_val_data = bin_data(children, val_data) if val_data else None
    return build_grids(children, children_data, t1, t2, children_val_data)

def bin_data(children, image_data):
    num_children = len(children)
    children_data = [[] for _ in range(num_children)]
    for img in image_data:
        for i in range(num_children):
            if children[i].contains(img.cell):
                children_data[i].append(img)
                break
    return children_data

def build_grids(children, children_data, t1, t2, children_val_data):
    ''' Goes through cells in children and further subdivides those grids. '''
    grids = dict()
    val_grids = dict()
    for i in range(len(children)):
        if (len(children_data[i]) < t2): continue
        if children_val_data is not None:
            child_grids, child_val_grids = _make_grids(children[i], children_data[i], t1, t2, children_val_data[i])
            val_grids.update(child_val_grids)
        else:
            child_grids, _ = _make_grids(children[i], children_data[i], t1, t2, None)
        grids.update(child_grids)
    return grids, val_grids

def latlon_to_cell(lat, lon):
    ''' Creates a s2 CellId from lat, lng coordinates '''
    latlng = s2.S2LatLng.FromDegrees(lat, lon)
    return s2.S2CellId.FromLatLng(latlng)

def get_latlng_from_token(token):
    cell_id = s2.S2CellId.FromToken(token)
    cell = s2.S2Cell(cell_id)
    latlng = s2.S2LatLng(cell.GetCenter())
    return latlng.lat().degrees(), latlng.lng().degrees()

class Image:
    def __init__(self, data):
        self.data = data
        self.lat = float(data[3])
        self.lng = float(data[2])
        self.cell = latlon_to_cell(self.lat, self.lng)

def save_grid(grid_list, grids, filename):
    ''' Saves grid metadata into filename '''
    f_out = open(filename, 'w')
    for i in range(len(grid_list)):
        cell = grid_list[i]
        data = grids[cell]
        token = cell.ToToken()
        for img in data:
            columns = img.data + [token, str(i)]
            f_out.write('\t'.join(columns) + '\n')
    f_out.close()

def process(train_file, t1, t2, prefix, val_file):
    ''' According to the PlaNET paper (https://arxiv.org/abs/1602.05314), we start from the six
    different quad tree roots and descend recursively subdividing each cell until no cell contains
    more than t1 photos, and then we discard cells with less than t2 photos.

    This function takes metadata from train_file to create grids using t1 and t2.
    '''
    image_data = open_dataset(train_file)
    val_data = None
    if val_file:
        val_data = open_dataset(val_file)

    grids, val_grids = make_grids(image_data, t1, t2, val_data)

    # Go through grids and write outputs
    grid_list = grids.keys()
    print("Grids made: {0}".format(len(grid_list)))
    size_list = [len(v) for v in grids.values()]
    train_stats = "Training - Max Grid Size: {0}, Min Grid Size: {1}, Median: {2}" \
                  .format(max(size_list), min(size_list), np.median(size_list))
    print(train_stats)
    if val_file:
        val_size_list = [len(v) for v in val_grids.values()]
        val_stats = "Validation - Max Grid Size: {0}, Min Grid Size: {1}, Median: {2}" \
                    .format(max(val_size_list), min(val_size_list), np.median(val_size_list))
        print(val_stats)

    # Precompute grid centers and centroids
    grid_info = []
    for cell in grids:
        data = grids[cell]
        token = cell.ToToken()
        center_lat, center_lng = get_latlng_from_token(token)
        # Calculate centroid
        centroid_lat = []
        centroid_lng = []
        for img in data:
            centroid_lat.append(img.lat)
            centroid_lng.append(img.lng)
        centroid_lat = np.mean(centroid_lat)
        centroid_lng = np.mean(centroid_lng)
        grid_info.append('\t'.join([token, str(center_lat), str(center_lng), str(centroid_lat), str(centroid_lng)]))

    pf = prefix + "_{0}-{1}-".format(str(t1), str(t2))
    with open(pf + "grids.txt", 'w') as f:
        f.write('\n'.join(grid_info))
    save_grid(grid_list, grids, pf + "data.train")
    if val_file:
        save_grid(grid_list, val_grids, pf + "data.val")
    with open(pf + "stats", 'w') as f:
        f.write("Num Grids: {0}\n".format(len(grid_list)))
        f.write(train_stats + "\n")
        if val_file: f.write(val_stats + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', help='Path to training data.')
    parser.add_argument('output_prefix')
    parser.add_argument('t1', type=int)
    parser.add_argument('t2', type=int)
    parser.add_argument('--val-path', help='Path to validation data.')
    args = parser.parse_args()
    process(args.data_path, args.t1, args.t2, args.output_prefix, args.val_path)
