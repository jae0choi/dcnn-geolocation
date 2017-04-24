import mxnet as mx
import logging

class SkippingIter(mx.io.DataIter):
    """Image data iterator with a large number of augumentation choices.
    Supports reading from both .rec files and raw image files with image list.
    To load from .rec files, please specify path_imgrec. Also specify path_imgidx
    to use data partition (for distributed training) or shuffling.
    To load from raw image files, specify path_imglist and path_root.
    Parameters
    ----------
    batch_size : int
        Number of examples per batch.
    data_shape : tuple
        Data shape in (channels, height, width).
        For now, only RGB image with 3 channels is supported.
    label_width : int
        dimension of label
    path_imgrec : str
        path to image record file (.rec).
        Created with tools/im2rec.py or bin/im2rec
    path_imglist : str
        path to image list (.lst)
        Created with tools/im2rec.py or with custom script.
        Format: index\t[one or more label separated by \t]\trelative_path_from_root.
    imglist: list
        a list of image with the label(s)
        each item is a list [imagelabel: float or list of float, imgpath].
    path_root : str
        Root folder of image files
    path_imgidx : str
        Path to image index file. Needed for partition and shuffling when using .rec source.
    shuffle : bool
        Whether to shuffle all images at the start of each iteration.
        Can be slow for HDD.
    part_index : int
        Partition index
    num_parts : int
        Total number of partitions.
    data_name : str
        data name for provided symbols
    label_name : str
        label name for provided symbols
    kwargs : ...
        More arguments for creating augumenter. See mx.image.CreateAugmenter.
    """

    def __init__(self, batch_size, data_shape, label_width=1,
                 path_imgrec=None, path_imglist=None, path_root=None, path_imgidx=None,
                 shuffle=False, part_index=0, num_parts=1, aug_list=None, imglist=None,
                 data_name='data', label_name='softmax_label', **kwargs):
        super(SkippingIter, self).__init__()
        assert path_imgrec or path_imglist or (isinstance(imglist, list))
        if path_imgrec:
            print('loading recordio...')
            if path_imgidx:
                self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r') # pylint: disable=redefined-variable-type
                self.imgidx = list(self.imgrec.keys)
            else:
                self.imgrec = mx.recordio.MXRecordIO(path_imgrec, 'r') # pylint: disable=redefined-variable-type
                self.imgidx = None
        else:
            self.imgrec = None

        if path_imglist:
            print('loading image list...')
            with open(path_imglist) as fin:
                imglist = {}
                imgkeys = []
                #x = 0
                for line in iter(fin.readline, ''):
                    #print(x)
                    #x += 1
                    line = line.strip().split('\t')
                    label = mx.nd.array([float(i) for i in line[1:-1]])
                    key = int(line[0])
                    imglist[key] = (label, line[-1])
                    imgkeys.append(key)
                self.imglist = imglist
        elif isinstance(imglist, list):
            print('loading image list...')
            result = {}
            imgkeys = []
            index = 1
            for img in imglist:
                key = str(index) # pylint: disable=redefined-variable-type
                index += 1
                if isinstance(img[0], numeric_types):
                    label = mx.nd.array([img[0]])
                else:
                    label = mx.nd.array(img[0])
                result[key] = (label, img[1])
                imgkeys.append(str(key))
            self.imglist = result
        else:
            self.imglist = None
        self.path_root = path_root

        self.check_data_shape(data_shape)
        self.provide_data = [(data_name, (batch_size,) + data_shape)]
        if label_width > 1:
            self.provide_label = [(label_name, (batch_size, label_width))]
        else:
            self.provide_label = [(label_name, (batch_size,))]
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.label_width = label_width

        self.shuffle = shuffle
        if self.imgrec is None:
            self.seq = imgkeys
        elif shuffle or num_parts > 1:
            assert self.imgidx is not None
            self.seq = self.imgidx
        else:
            self.seq = self.imgidx

        if num_parts > 1:
            assert part_index < num_parts
            N = len(self.seq)
            C = N/num_parts
            self.seq = self.seq[part_index*C:(part_index+1)*C]
        if aug_list is None:
            self.auglist = mx.image.CreateAugmenter(data_shape, **kwargs)
        else:
            self.auglist = aug_list
        self.cur = 0
        self.reset()

    def reset(self):
        if self.shuffle:
            random.shuffle(self.seq)
        if self.imgrec is not None:
            self.imgrec.reset()
        self.cur = 0

    def next_sample(self):
        """Helper function for reading in next sample."""
        if self.seq is not None:
            if self.cur >= len(self.seq):
                raise StopIteration
            idx = self.seq[self.cur]
            self.cur += 1
            if self.imgrec is not None:
                s = self.imgrec.read_idx(idx)
                header, img = mx.recordio.unpack(s)
                if self.imglist is None:
                    return header.label, img
                else:
                    return self.imglist[idx][0], img
            else:
                label, fname = self.imglist[idx]
                return label, self.read_image(fname)
        else:
            s = self.imgrec.read()
            if s is None:
                raise StopIteration
            header, img = mx.recordio.unpack(s)
            return header.label, img

    def next(self):
        batch_size = self.batch_size
        c, h, w = self.data_shape
        batch_data = mx.nd.empty((batch_size, c, h, w))
        batch_label = mx.nd.empty(self.provide_label[0][1])
        i = 0
        skipped = 0
        try:
            while i < batch_size:
                #if i % 100 == 0: logging.info("Loaded {}, skipped {}".format(i, skipped))
                label, s = self.next_sample()
                #logging.info(label)
                if label.asnumpy()[0] == -1.0: 
                    #logging.info("SKIPPING")
                    #skipped += 1
                    continue
                data = [self.imdecode(s)]
                try:
                    self.check_valid_image(data)
                except RuntimeError as e:
                    logging.debug('Invalid image, skipping:  %s', str(e))
                    continue
                data = self.augmentation_transform(data)
                for datum in data:
                    assert i < batch_size, 'Batch size must be multiples of augmenter output length'
                    batch_data[i][:] = self.postprocess_data(datum)
                    batch_label[i][:] = label
                    i += 1
        except StopIteration:
            if not i:
                raise StopIteration
        #logging.info("Skipped: {}".format(skipped))
        return mx.io.DataBatch([batch_data], [batch_label], batch_size-i)

    def check_data_shape(self, data_shape):
        """checks that the input data shape is valid"""
        if not len(data_shape) == 3:
            raise ValueError('data_shape should have length 3, with dimensions CxHxW')
        if not data_shape[0] == 3:
            raise ValueError('This iterator expects inputs to have 3 channels.')

    def check_valid_image(self, data):
        """checks that data is valid"""
        if len(data[0].shape) == 0:
            raise RuntimeError('Data shape is wrong')

    def imdecode(self, s):
        """decodes a sting or byte string into an image."""
        return mx.image.imdecode(s)

    def read_image(self, fname):
        """reads image from fname and returns the raw bytes to be decoded."""
        with open(os.path.join(self.path_root, fname), 'rb') as fin:
            img = fin.read()
        return img

    def augmentation_transform(self, data):
        """transforms data with specificied augmentation."""
        for aug in self.auglist:
            data = [ret for src in data for ret in aug(src)]
        return data

    def postprocess_data(self, datum):
        """final postprocessing step before image is loaded into the batch."""
        return mx.nd.transpose(datum, axes=(2, 0, 1))
