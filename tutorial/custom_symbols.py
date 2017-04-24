import mxnet as mx
import numpy as np

# TODO: Hopefully find a better way than to have a hardcoded path to the grid data.
GRID_FILE = "data/yfcc100m__5000-500-grids.txt"

class L2GeolocationLoss(mx.operator.CustomOp):
    # Save a bunch of ndarrays and use concatenate to generate labels for grids?
    def __init__(self):
        self.grid_latlngs = []
        with open(GRID_FILE, 'r') as f:
            for i, line in enumerate(f):
                line = line.split('\t')
                lat = float(line[3])
                lng = float(line[4])
                self.grid_latlngs.append(mx.nd.array([[lat, lng]]))

    def forward(self, is_train, req, in_data, out_data, aux):
        # Compute softmax function because want to be able to get accuracy.
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        labels = in_data[1].asnumpy().ravel().astype(np.int)
        pred = mx.nd.argmax(out_data[0], 1).asnumpy().astype(np.int)
        
        # Compute L2 Distance between highest confidence prediction and ground truth.
        label_latlng = mx.nd.concatenate([self.grid_latlngs[int(i)] for i in labels]).asnumpy()
        pred_latlng = mx.nd.concatenate([self.grid_latlngs[int(i)] for i in pred]).asnumpy()
        dist1 = pred_latlng - label_latlng
        dist2 = np.array([180, 360]) - dist1
        dist = np.minimum(dist1, dist2)

        # Set the gradient as -1.0 * distance in the predicted location.
        l2 = np.log(1.0 + np.sum(dist ** 2, 1))
        out = in_grad[0].asnumpy()
        out[np.arange(pred.shape[0]), pred.astype(np.int)] = -1.0
        out = out * l2[:, None]
        self.assign(in_grad[0], req[0], mx.nd.array(out))

@mx.operator.register("L2GeolocationLoss")
class L2GeolocationLossProp(mx.operator.CustomOpProp):
    ''' Uses L2 Loss with latlngs labels to train.
    To use, register symbol as:
        mx.symbol.Custom(data="fc1", name='l2loss', op_type="L2GeolocationLoss")
    '''
    def __init__(self):
        super(L2GeolocationLossProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return L2GeolocationLoss()

class L2WeightedSoftmax(mx.operator.CustomOp):
    # Save a bunch of ndarrays and use concatenate to generate labels for grids?
    def __init__(self):
        self.grid_latlngs = []
        with open(GRID_FILE, 'r') as f:
            for i, line in enumerate(f):
                line = line.split('\t')
                lat = float(line[3])
                lng = float(line[4])
                self.grid_latlngs.append(mx.nd.array([[lat, lng]]))

    def forward(self, is_train, req, in_data, out_data, aux):
        # Compute softmax function because want to be able to get accuracy.
        x = in_data[0].asnumpy()
        y = np.exp(x - x.max(axis=1).reshape((x.shape[0], 1)))
        y /= y.sum(axis=1).reshape((x.shape[0], 1))
        self.assign(out_data[0], req[0], mx.nd.array(y))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        labels = in_data[1].asnumpy().ravel().astype(np.int)
        pred = mx.nd.argmax(out_data[0], 1).asnumpy().astype(np.int)

        # Compute log L2 distance between highest confidence prediction and ground truth.
        label_latlng = mx.nd.concatenate([self.grid_latlngs[int(i)] for i in labels]).asnumpy()
        pred_latlng = mx.nd.concatenate([self.grid_latlngs[int(i)] for i in pred]).asnumpy()
        dist1 = pred_latlng - label_latlng
        dist2 = np.array([180, 360]) - dist1
        dist = np.minimum(dist1, dist2)
        l2 = np.log(1.0 + np.sum(dist ** 2, 1))

        # Weight the cross-entropy loss by the previously calculated log L2 distance.
        y = out_data[0].asnumpy()
        y[np.arange(labels.shape[0]), labels] -= 1.0
        y = y * l2[:, None]
        self.assign(in_grad[0], req[0], mx.nd.array(y))

@mx.operator.register("L2WeightedSoftmax")
class L2WeightedSoftmaxProp(mx.operator.CustomOpProp):
    ''' Uses L2 Loss with latlngs labels to train.
    To use, register symbol as:
        mx.symbol.Custom(data="fc1", name='l2loss', op_type="L2GeolocationLoss")
    '''
    def __init__(self):
        super(L2WeightedSoftmaxProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data', 'label']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return L2WeightedSoftmax()
