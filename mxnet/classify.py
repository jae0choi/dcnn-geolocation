import argparse
import time
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

import mxnet as mx
import numpy as np
import custom_symbols

def get_test_iter(test_data, test_labels, batch_size):
    mean = np.array([123.68, 116.28, 103.53])
    test_iter = mx.io.ImageRecordIter(
        path_imglist        = test_labels,
        path_imgrec         = test_data,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = (3, 224, 224),
        mean_r              = mean[0],
        mean_g              = mean[1],
        mean_b              = mean[2])
    return test_iter

def correct_pred(top5, label):
    c1 = int(top5[0] == label)
    c5 = 0
    for p in top5:
        if p == label: c5 = 1
    return c1, c5

def classify(test_data, saved_model, epoch, save_name=None, batch_size=200, num_gpus=0,
             test_labels="", no_acc=True, short=False):
    batch_size = max(batch_size, num_gpus * batch_size)
    test_iter = get_test_iter(test_data, test_labels, batch_size)

    # Load saved model.
    sym, arg_params, aux_params = mx.model.load_checkpoint(saved_model, epoch)
    devs = mx.cpu()
    if num_gpus > 0:
        devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label, for_training=False)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    if save_name:
        postfix = ""
        if short: postfix = ".short"
        f_out = open(save_name + ".pred" + postfix, 'w')
        info_out = open(save_name + ".info" + postfix, 'w')
        info_out.write("saved_model: {}, test_data: {}\n".format("%s-%04d" % (saved_model, epoch), test_data))
    total, top1_correct, top5_correct = 0.0, 0.0, 0.0
    num_in_time = 0
    tic = time.time()
    for pred, i_batch, data_batch in mod.iter_predict(test_iter):
        if i_batch % 10 == 0 and i_batch != 0:
            sps = float(num_in_time)/(time.time() - tic)
            logging.info("Batch:%d\t%f samples/sec" % (i_batch, sps))
            num_in_time = 0
            tic = time.time()
        labels = None
        if not no_acc:
            labels = data_batch.label[0].asnumpy()
        index = data_batch.index
        pred = pred[0].asnumpy()
        num = pred.shape[0]
        total += num
        num_in_time += num
        for i in range(num):
            prob = pred[i]
            top5 = np.argsort(prob)[::-1][0:5]
            if not no_acc:
                c1, c5 = correct_pred(top5, labels[i])
                top1_correct += c1
                top5_correct += c5
            if save_name:
                results = ["%d\t%f" % (p, prob[p]) for p in top5]
                f_out.write("{}\t{}\n".format(index[i], '\t'.join(results)))
        if short and total >= 100000:
            break
    acc_str = "no_acc: True"
    if not no_acc:
        acc_str = "Top1 Acc: %f\tTop5 Acc: %f" % (top1_correct/total, top5_correct/total)
    print(acc_str)
    if save_name: 
        info_out.write(acc_str)
    return acc_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data', help='Path to training data.')
    parser.add_argument('saved_model', help='Saved model weights')
    parser.add_argument('epoch', type=int, help='Which training epoch to use')
    parser.add_argument('--save-name', help='Name to save results as.')
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=200)
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--test-labels', default='')
    parser.add_argument('--no-acc', action="store_true", help="Set flag to not test accuracy")
    parser.add_argument('--short', action="store_true", help="Classify only first 100k of testset")
    args = parser.parse_args()

    classify(test_data=args.test_data,
             saved_model=args.saved_model,
             epoch=args.epoch,
             save_name=args.save_name,
             batch_size=args.batch_size,
             num_gpus=args.num_gpus,
             test_labels=args.test_labels,
             no_acc=args.no_acc,
             short=args.short)
