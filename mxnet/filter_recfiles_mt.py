import argparse
import multiprocessing
import time

import mxnet as mx

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("srcs", help="Comma-separated list of recfile prefix to retrieve images from")
    parser.add_argument("imglist", help="Text file of new images to include, should be formatted img\\tlabel\\n")
    parser.add_argument("savename", help="Prefix for new recfile")
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--s3", help="Use recfile from this s3 bucket.", default="")
    
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
    
    if args.num_threads > 1:
        def read_worker(record, q_in, q_out):
            while True:
                deq = q_in.get()
                if deq is None:
                    break
                i, idx = deq
                s = record.read_idx(idx)
                q_out.put((idx,s))
        def write_worker(record, q_out):
            global cur_idx, tic
            more = True
            while more:
                deq = q_out.get()
                if deq is not None:
                    idx, s = deq
                else:
                    more = False
                header, img = mx.recordio.unpack_img(s)
                imgname = indices[idx]
                label = imgset[imgname]
                new_header = mx.recordio.IRHeader(0, label, cur_idx, 0)
                buf = mx.recordio.pack_img(new_header, img, quality=100)
                record.write_idx(cur_idx, buf)
                lst_out.write("{}\t{}\t{}\n".format(cur_idx, label, imgname))
                cur_idx += 1
                if cur_idx % 100 == 0:
                    print((cur_idx, time.time() - tic))
                    tic = time.time()
        for recname in recfiles:
            q_in = [multiprocessing.Queue(1024) for i in range(args.num_threads)]
            q_out = multiprocessing.Queue(1024)
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
            record = [mx.recordio.MXIndexedRecordIO(recname + '.idx', args.s3 + recname + '.rec', 'r') for _ in range(args.num_threads)]
            idxs = sorted(list(indices.keys()))
            read_process = [multiprocessing.Process(target=read_worker, args=(record[i], q_in[i], q_out)) \
                    for i in range(args.num_threads)]
            for p in read_process:
                p.start()
            write_process = multiprocessing.Process(target=write_worker, args=(new_record, q_out))
            write_process.start()
            for i, item in enumerate(idxs):
                q_in[i % len(q_in)].put((i, item))
            for q in q_in:
                q.put(None)
            for p in read_process:
                p.join()
            
            q_out.put(None)
            write_process.join()
    else:
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
                if cur_idx % 100 == 0:
                    print((cur_idx, time.time() - tic))
                    tic = time.time()

        for recname in recfiles:
            filter_record(recname)


