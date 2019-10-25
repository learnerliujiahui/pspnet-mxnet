import sys
sys.path.append('..')
import argparse
import timeit
import numpy as np
import mxnet as mx
from mxnet import nd, gluon, autograd, init
from mxnet import gpu
from mxnet import cpu
from mxnet.gluon.model_zoo import vision as models
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data.vision.transforms import RandomFlipTopBottom, ToTensor, Normalize, RandomFlipLeftRight
from dataloaders.dataset import ISPRSDataset, ISPRSSegmentation
from dataloaders.transforms import RandomCrop, ToTensor, FixedResize,RandomRotate, RandomHorizontalFlip, Normalize

from models.modules import ResNet18
import utils
parser = argparse.ArgumentParser('Parameters for PSPNet of Mxnet version')
parser.add_argument('--mode', default='train')
parser.add_argument('--data', default='ISPRS')
parser.add_argument('--num_classes', default=6)
parser.add_argument('--data_dir', default='/home/liujiahui/data_zoo/VOCdevkit/VOC2012')
parser.add_argument('--epochs', default=50)
parser.add_argument('--crop_size', default=256)
parser.add_argument('--num_gpus', default=1)
parser.add_argument('--init_lr', default=0.02)
parser.add_argument('--weight_decay', default=0.001)
parser.add_argument('--batch_size', default=1)

args = parser.parse_args()

# show voc2012 image
# utils.show_image(imgs,nrows=3, ncols=2,figsize=(12,8))


if __name__ == '__main__':
    # set devices
    ctx = utils.try_all_gpus()

    # pixel mean and std for ImageNet data
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_augs = transforms.Compose([
                                    RandomHorizontalFlip(),
                                    RandomCrop(args.crop_size),
                                    RandomRotate(15),
                                    Normalize(mean=mean, std=std),
                                    ToTensor(),
                                    ])

    test_augs = transforms.Compose([
                                    FixedResize(size=(512, 512)),
                                    Normalize(mean=mean, std=std),
                                    ToTensor(),
                                    ])

    #
    isprs_train = ISPRSDataset(is_train=True, transform=train_augs)
    isprs_test = ISPRSDataset(is_train=False, transform=None)

    # generate the dataloader
    trainloader = gluon.data.DataLoader(isprs_train, batch_size=args.batch_size, shuffle=True, last_batch='discard')
    testloader = gluon.data.DataLoader(isprs_test, batch_size=1, last_batch='discard')

    print(len(trainloader))
    print(len(testloader))


    # set the model info
    pretrained_model = models.resnet18_v2(pretrained=True, ctx=ctx)
    model = nn.HybridSequential()
    # build the layer
    for layer in pretrained_model.features[:-2]:
        model.add(layer)

    with model.name_scope():
        model.add(
            nn.Conv2D(args.num_classes, kernel_size=1),
            nn.Conv2DTranspose(args.num_classes, kernel_size=64, padding=16, strides=32)
        )

    model[-2].initialize(init=init.Xavier(), ctx=ctx)
    model[-1].initialize(init=init.Constant(utils.bilinear_kernel(args.num_classes, args.num_classes, 64)), ctx=ctx)
    # set the loss function
    loss = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)

    model.collect_params().reset_ctx(ctx=ctx)
    trainer =gluon.Trainer(
                           params=model.collect_params(),
                           optimizer='sgd',
                           optimizer_params={'learning_rate': args.init_lr, 'wd': args.weight_decay},
                           )

    print("Start training...")
    for epoch in range(args.epochs):
        start_time = timeit.default_timer()
        # setting the learning rate
        lr_ = utils.lr_poly(args.init_lr, epoch, args.epochs, 0.9)

        train_l_sum, train_acc_sum, n, m, start = 0.0, 0.0, 0, 0, timeit.default_timer()

        for i, batch in enumerate(trainloader):
            imgBatchList, lblBatchList, batch_size = utils._get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [model(img) for img in imgBatchList]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, lblBatchList)]

            for l in ls:
                l.backward()

            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_acc_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, lblBatchList)])
        m += sum([y.size for y in lblBatchList])
        test_acc = utils.evaluate_accuracy(testloader, model, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f time %.1f sec' % (epoch + 1, train_l_sum / n, train_acc_sum / m, test_acc, timeit.default_timer()- start))



# # for cifar10 dataset --> num_classes=10
# net = ResNet18(num_classes=10)
# net.initialize(init=init.Xavier(), ctx=ctx)
#
# loss = gluon.loss.SoftmaxCrossEntropyLoss()
# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning rate': args.init_lr})
#
# for epoch in range(args.epoches):
#     start = time()
#     total_loss = 0
#     for data, label in train_data:
#         data_list = gluon.utils.split_and_load(data, ctx_list=ctx)
#         label_list = gluon.utils.split_and_load(label, ctx_list=ctx)
#         with autograd.record():
#             losses = [loss(net(x), y) for x,y in zip(data_list, label_list)]
#         for l in losses:
#             l.backward()
#         total_loss += sum([l.sum().asscalar() for l in losses])
#         trainer.step(args.batch_size)
#
#     nd.waitall()
#     print('Epoch %d, training time = %.1f sec' % (epoch+1, time() - start))
#
# # x = nd.random.normal(shape=(4, 1, 28, 28))
# # x_list = gluon.utils.split_and_load(x, ctx)
# weight = net[1].params.get('weight')
# print(weight.data(ctx[0])[0])
