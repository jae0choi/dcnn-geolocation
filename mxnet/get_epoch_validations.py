import mxnet as mx
import numpy as np

from classify import classify

val_data = "../../../../yfcc100m/recfiles/yfcc100m-5000-500-val.rec"
saved_model = "places365-5000-500"
prefix_change = 8

f_out = open("train_vals", 'a')
for i in range(1, 23):
    if i == prefix_change: saved_model = "2-" + saved_model
    acc = classify(val_data, "models/" + saved_model, i, save_name=None, batch_size=200, num_gpus=8, test_labels="", no_acc=False)
    print(acc)
    f_out.write("{}:\t{}\n".format(i, acc))
