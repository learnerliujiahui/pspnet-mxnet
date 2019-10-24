import sys
sys.path.append('..')
import argparse
import time
import mxnet as mx
from mxnet import nd, gluon, autograd, init
from mxnet import gpu
from mxnet import cpu
from mxnet.gluon.model_zoo import vision as models
from models.modules import ResNet18
import utils
parser = argparse.ArgumentParser('Parameters for PSPNet of Mxnet version')
parser.add_argument('--mode', default='train')
parser.add_argument('--data', default='ISPRS')
parser.add_argument('--data_dir', default='/home/liujiahui/data_zoo/VOCdevkit/VOC2012')
parser.add_argument('--epoches', default=50)
parser.add_argument('--num_gpus', default=4)
parser.add_argument('--init_lr', default=0.02)
parser.add_argument('--batch_size', default=64)

args = parser.parse_args()


ctx = [gpu(i) for i in range(args.num_gpus)]

train_images, train_labels = utils.read_images(root=args.data_dir, train=True)
imgs = []
for i in range(3):
    imgs += [train_images[i], train_labels[i]]

# show voc2012 image
# utils.show_image(imgs,nrows=3, ncols=2,figsize=(12,8))

y = utils.image2label(train_labels[0])
print(y[105:115, 130:140])

mean =nd.array([0.485, 0.456, 0.406])
std =nd.array([0.229, 0.224, 0.225])

def normalize(data):
    return (data.astype('float32') / 255 - mean) / std

class VOCSegDataset(gluon.data.Dataset):
    def __init__(self, is_train, crop_size):
        self.crop_size = crop_size
        data, label = utils.read_images(train=is_train)
        data = self._filter(data)
        print(len(data))
        self.data = [normalize(img) for img in data]
        self.label = self._filter(label)

        print(len(label))
        assert len(self.data) == len(self.label)
        print('Read '+str(len(self.data))+' examples')

    def __getitem__(self, item):
        data, label = utils.RandomCrop(self.data[item], self.label[item], *self.crop_size)
        data = data.transpose((2, 0, 1))
        return data, label

    def __len__(self):
        return len(self.data)

    def _filter(self, images):
        '''only keep the images, which size larger than the crop size'''
        return [img for img in images if img.shape[0] > self.crop_size[0] and img.shape[1] > self.crop_size[1]]

input_shape = (320, 480)
voc_train = VOCSegDataset(is_train=True,crop_size=input_shape)
voc_test = VOCSegDataset(is_train=False,crop_size=input_shape)

train_data = gluon.data.Dataloader(voc_train, args.batch_size, shuffle=True, last_batch='discard')
test_data = gluon.data.Dataloader(voc_test, batch_size=1, last_batch='discard')

pretrained_mdoel = models.resnet18_v2(pretrained=True)

model = nn.HybridSequential()
# build the layer
for layer in pretrained_model[:-2]:
    net.add(layer)

print(mdoel)
num_classes = 21

with net.name_scope():
    net.add(
            nn.Conv2D(num_classes, kernel_size=1),
            nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16, strides=32)
           )

# set devices
ctx = utils.try_all_gpus()
# loss function
loss = gloss.SoftmaxCrossEntropyLoss(axis=1)

net.collect_params().reset_ctx(ctx)
# set optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1, 'wd': 1e-3})
# start training
utils.train(train_data, 
            test_data, net, loss, trainer, ctx, 
            num_epochs=args.epochs)



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
