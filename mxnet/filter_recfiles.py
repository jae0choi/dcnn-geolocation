import argparse
import time

import mxnet as mx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srcs", help="Comma-separated list of recfile prefix to retrieve images from")
    parser.add_argument("imglist", help="Text file of new images to include, should be formatted img\\tlabel\\n")
    parser.add_argument("savename", help="Prefix for new recfile")
    parser.add_argument("--s3", help="Use recfile from this s3 bucket.")
    
    args = parser.parse_args()    
    recfiles = args.srcs.split(',')
    img_list = args.imglist
    new_prefix = args.savename

    cur_idx = 0
    lst_out = open(new_prefix + '.lst', 'w')
    new_record = mx.recordio.MXIndexedRecordIO(new_prefix + '.idx', new_prefix + '.rec', 'w')
    imgset = dict()
    with open(img_list, 'r') as f:
        for line in f:
            line = line.split('\t')
            imgset[line[0]] = float(line[1])

    tic = time.time()

    def filter_record(recname):
        global cur_idx, tic
        indices = dict()
        available_indices = set()
        with open(recname + '.idx', 'r') as f:
            for line in f:
                available_indices.add(int(line.split('\t')[0]))
        with open(recname + '.lst', 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                imgname = line[-1]
                idx = int(line[0])
                if imgname in imgset and idx in available_indices:
                    indices[idx] = imgname
        record = mx.recordio.MXIndexedRecordIO(recname + '.idx', args.s3 + recname + '.rec', 'r')
        idxs = sorted(list(indices.keys()))
        for idx in idxs:
            imgname = indices[idx]
            s = record.read_idx(idx)
            header, img = mx.recordio.unpack_img(s)
            label = imgset[imgname]
            new_header = mx.recordio.IRHeader(0, label, cur_idx, 0)
            buf = mx.recordio.pack_img(new_header, img, quality=100) #Preserve image quality because rewriting.
            new_record.write_idx(cur_idx, buf)
            lst_out.write("{}\t{}\t{}\n".format(cur_idx, label, imgname))
            cur_idx += 1
            if cur_idx % 1000 == 0:
                print((cur_idx, time.time() - tic))
                tic = time.time()

    for recname in recfiles:
        filter_record(recname)


