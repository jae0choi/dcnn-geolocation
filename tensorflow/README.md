# Deep Learning for Geolocation with Tensorflow
This repo contains the code used for training deep convolutional neural nets on large image datasets
for geolocation based on Google's [PlaNet Paper](https://arxiv.org/abs/1602.05314).

## Usage
### Requirements
1. Install the [S2 Geometry Library](https://github.com/micolous/s2-geometry-library) which is
required for dividing training data into grids.
2. Install [tensorflow](https://www.tensorflow.org/).
3. Use [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow) to convert a pre-trained
Caffe model to tensorflow.

### Training and Classification
1. Use `make_grids.py` to divide your training (and validation) data into classes.
2. With your converted model or the provided `google_net.py`, modify and use `train.py` to train
your model.
3. Use `classify.py` to predict geolocation for your test images!

## Notes
A big thanks to user ethereon for their
[caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow)
library. Their code was used to convert Caffe models as well as provide some core functionality for
these training scripts.
