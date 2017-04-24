import argparse
import time
import logging
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

import mxnet as mx
import numpy as np
import custom_symbols

def get_test_iter(args, batch_size):
    mean = np.array([float(value) for value in args.rgb_mean.split(',')])
    test_iter = mx.io.ImageRecordIter(
        path_imglist        = args.test_labels,
        path_imgrec         = args.test_data,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = (3, 224, 224),
        mean_r              = mean[0],
        mean_g              = mean[1],
        mean_b              = mean[2])
    return test_iter

def correct_pred(top5, label):
    c1 = int(top5[0] == label) # 1 if Top 1 prediction correct.
    c5 = sum([p == label for p in top 5]) # 1 if Top 5 prediction correct.
    return c1, c5

def classify(args):
    batch_size = max(args.batch_size, args.num_gpus * args.batch_size)
    test_iter = get_test_iter(args, batch_size)

    # Load saved model.
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.saved_model, args.epoch)
    devs = mx.cpu()
    if args.num_gpus > 0:
        devs = [mx.gpu(i) for i in range(args.num_gpus)]
    mod = mx.mod.Module(symbol=sym, context=devs)
    mod.bind(data_shapes=test_iter.provide_data, label_shapes=test_iter.provide_label, for_training=False)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # Create save files and write out info about model and test data.
    postfix = ""
    if args.short: postfix = ".short"
    f_out = open(args.save_name + ".pred" + postfix, 'w')
    info_out = open(args.save_name + ".info" + postfix, 'w')
    info_out.write("saved_model: {}, test_data: {}\n".format("%s-%04d" % (args.saved_model, args.epoch), args.test_data))

    # Iterate through the test iter and save accuracy and predictions.
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
        if not args.no_acc:
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
            results = ["%d\t%f" % (p, prob[p]) for p in top5]
            f_out.write("{}\t{}\n".format(index[i], '\t'.join(results)))
        if args.short and total >= 100000:
            break
    acc_str = "no_acc: True"
    if not args.no_acc:
        acc_str = "Top1 Acc: %f\tTop5 Acc: %f" % (top1_correct/total, top5_correct/total)
    print(acc_str)
    info_out.write(acc_str)
    return acc_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('test_data', help='Path to training data.')
    parser.add_argument('saved_model', help='Saved model weights')
    parser.add_argument('epoch', type=int, help='Which training epoch to use')
    parser.add_argument('save_name', help='Name to save results as.')
    parser.add_argument('--batch-size', help='Batch size.', type=int, default=200)
    parser.add_argument('--num-gpus', type=int, default=0)
    parser.add_argument('--rgb-mean', type=str, default='123.68,116.779,103.939',
                        help='a tuple of size 3 for the mean rgb')
    parser.add_argument('--test-labels', default='')
    parser.add_argument('--no-acc', action="store_true", help="Set flag to not test accuracy")
    parser.add_argument('--short', action="store_true", help="Classify only first 100k of testset")
    args = parser.parse_args()

    classify(args)
