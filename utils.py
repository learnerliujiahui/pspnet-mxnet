import time
import random
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, image

import matplotlib.pyplot as plt


# Batch Normalization
def pure_batch_norm(x, gamma, beta, eps=1e-5):
    assert len(x.shape) in (2, 4)
    if len(x.shape) == 2:  # fc layer: batch * feature
        mean = x.mean(axis=0)
        variance = ((x - mean) ** 2).mean(axis=0)
    else:
        # 2D Tensor: batch * channel * height * width
        mean = x.mean(axis=(0, 2, 3), keepdim=True)
        variance = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdim=True)

    x_hat = (x - mean) / nd.sqrt(variance + eps)
    return gamma.reshape(mean.shape) * x_hat + beta.reshape(mean.shape)


def batch_norm(x, gamma, beta, is_training, moving_mean, moving_variance, eps=1e-5, moving_momentum=0.9):
    assert len(x.shape) in (2, 4)
    if len(x.shape) == 2:
        mean = x.mean(axis=0)
        variance = ((x - mean) ** 2).mean(axis=0)
    else:
        mean = x.mean(axis=(0, 2, 3), keepdim=True)
        variance = ((x - mean) ** 2).mean(axis=(0, 2, 3), keepdim=True)
        # make sure the boardcasting machism
        moving_mean = moving_mean.reshape(mean.shape)
        moving_variance = moving_variance.reshape(mean.shape)
    if is_training:
        x_hat = (x - mean) / nd.sqrt(variance + eps)
        # update the global mean and variance
        moving_mean[:] = moving_momentum * moving_mean + (1.0 - moving_momentum) * mean
        moving_variance[:] = moving_momentum * moving_variance + (1.0 - moving_momentum) * variance
    else:
        # testing: using the training stage mean and variance
        x_hat = (x - moving_mean) / nd.sqrt(moving_variance + eps)

    return gamma.reshape(mean.shape) * x_hat + beta.reshape(mean.shape)


# data augmentation
def get_transform(augs):
    def transform(data, label):
        data = data.astype('float32')
        if augs is not None:
            data = apply_aug_list(data, augs)
        data = nd.transpose(data, (2, 0, 1)) / 255
        return data, label.astype('float32')


# multi-GPUs
def get_params(params, ctx):
    new_params = [p.copyto(ctx) for p in params]
    for p in new_params:
        p.attach_grad()
    return new_params


def allreduce(data):
    """
    assemble data from all gpus, and then allocate to itself
    :param data:
    :return:
    """
    # sum on data[0].context, and then broadcast
    for i in range(1, len(data)):
        data[0][:] += data[i].copyto(data[0].context)
    for i in range(1, len(data)):
        data[0].copyto(data[i])


def data_iter(batch_size, num_examples, data, label):
    idx = list(range(num_examples))
    random.shuffle(idx)
    for batch_i, i in enumerate(range(0, num_examples, batch_size)):
        batch = nd.array(idx[i:min(i + batch_size, num_examples)])
        yield batch_i, data.take(batch), label.take(batch)

def square_loss(yhat, y):
    return (yhat - y.reshape(yhat.shape)) **2 /2

def SGD(params, lr, batch_size):
    """

    :param params: model parameters
    :param lr: learning rate
    :return:
    """
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def try_all_gpus():
    ctxes = []
    try:
        for i in range(4):
            # 假设一台机器上GPU的数量不超过16
            ctx = mx.gpu(i)
            _ = nd.array([0], ctx=ctx)
            ctxes.append(ctx)
    except mx.base.MXNetError:
        pass
    if not ctxes:
        ctxes = [mx.cpu()]
    return ctxes

def _get_batch(batch, ctx):
    features, labels = batch
    if labels.dtype != features.dtype:
        labels = labels.astype(features.dtype)
    return (split_and_load(features, ctx),
            split_and_load(labels, ctx),
            features.shape[0])

def split_and_load(data, ctx):
    n, k = data.shape[0], len(ctx)
    m = n // k
    assert m * k == n, '# examples is not divided by # devices.'
    return [data[i * m: (i + 1) * m].as_in_context(ctx[i]) for i in range(k)]


def evaluate_accuracy(data_iter, net, ctx=[mx.cpu()]):
    if isinstance(ctx, mx.Context):
        ctx = [ctx]
    acc_sum, n = nd.array([0]), 0
    for batch in data_iter:
        features, labels, _ = _get_batch(batch, ctx)
        for X, y in zip(features, labels):
            y = y.astype('float32')
            acc_sum += (net(X).argmax(axis=1) == y).sum().copyto(mx.cpu())
            n += y.size
        acc_sum.wait_to_read()
    return acc_sum.asscalar() / n


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('training on', ctx)

    if isinstance(ctx, mx.Context):
        ctx = [ctx]

    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
        m += sum([y.size for y in ys])
    test_acc = evaluate_accuracy(test_iter, net, ctx)
    print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f time %.1f sec' % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc,
                                                                                time.time() - start))

voc_root = '/home/liujiahui/data_zoo/VOCdevkit/VOC2012'
def read_images(root=voc_root, train=True):
    txt_filename = root + '/ImageSets/Segmentation/' + ('train.txt' if train else 'val.txt')
    with open(txt_filename, 'r') as f:
        images = f.read().split()
    n = len(images)
    data, label = [None]*n, [None]*n
    for i, filename in enumerate(images):
        data[i] = image.imread('%s/JPEGImages/%s.jpg' % (root, filename))
        label[i] = image.imread('%s/SegmentationClass/%s.png' % (root, filename))
    return data, label


def show_image(imgs, nrows, ncols, figsize=None):
    if not figsize:
        figsize = (ncols, nrows)
    _, figs = plt.subplots(nrows, ncols, figsize=figsize)
    for i in range(nrows):
        for j in range(ncols):
            figs[i][j].imshow(imgs[i*ncols + j].asnumpy())
            figs[i][j].axes.get_xaxis().set_visible(False)
            figs[i][j].axes.get_yaxis().set_visible(False)
    plt.show()


def RandomCrop(data, label, height, width):
    data, rect = image.random_crop(data,(height, width))
    label = image.fixed_crop(label,*rect)
    return data, label




# using bilinear kernel initialize conv2dtranspose
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    weight[range(in_channels), range(out_channels), :, :] = filt
    return nd.array(weight)


def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)