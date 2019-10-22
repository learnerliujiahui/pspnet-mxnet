from mxnet.gluon import nn
from mxnet import autograd, gluon, nd, init

batch_size = 10

ctx = utils.try_gpu()
# initialization method: init.Xavier()
net.initialize(ctx=ctx, init=init.Xavier())

sotfmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.901})


# AlexNet structure
net = nn.Sequential()
with net.name_scope():
    # the first stage
    net.add(nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=3, strides=2))

    # the second stage
    net.add(nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=3, strides=2))
    
    # the third stage
    net.add(nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
    net.add(nn.MaxPool2D(pool_size=3, strides=2))

    # 4th stage
    net.add(nn.Flatten())
    net.add(nn.Dense(4096, activation='relu'))
    net.add(nn.Dropout(0.5))
    
    # 5th stage
    net.add(nn.Dense(4096, activation='relu'))
    net.add(nn.Dropout(0.5))
    
    # 6th stage
    net.add(nn.Dense(10))

# prepare the dataset
epoches = 10

for epoch in range(epoches):
    train_loss = 0
    train_acc = 0
    for data, label in train_data:
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data.as_in_context(ctx))
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss = nd.mean(loss).asscalar()
        train_acc = utils.accuarcy(output , label)


# vgg block
def vgg_block(num_convs, channels):
    out = nn.Sequential():
    with out.name_scope():
        for _ in range(num_convs):
            out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))
        out.add(nn.MaxPool2D(pool_size=2, strides=2))
    return out

# VggNet

conv_arch = ((1, 64),(1, 128),(2, 256),(2, 512),(2, 512))

def vgg(conv_arch):
    net = nn.Sequential()
    for (num_covns, num_channels) in conv_arch:
        net.add(vgg_block(nunm_convs, num_channels))
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10)
    )
net = vgg(conv_arch)


# Batch Normalization
def pure_batch_norm(x, gamma, beta, eps=1e-5):
    assert len(x.shape) in (2, 4)
    if len(x.shape) == 2:  # fc layer: batch * feature
        mean = x.mean(axis=0)
        variance = ((x - mean)**2).mean(axis=0)
    else:
        # 2D Tensor: batch * channel * height * width
        mean = x.mean(axis=(0, 2, 3), keepdim=True)
        variance = ((x - mean)**2).mean(axis=(0, 2, 3), keepdim=True)

    x_hat = (x - mean) / nd.sqrt(variance + eps)
    return gamma.reshape(mean.shape) * x_hat + beta.reshape(mean.shape)


def batch_norm(x, gamma, beta, is_training, moving_mean, moving_variance, eps=1e-5, moving_momentum=0.9):
    assert len(x.shape) in (2, 4)
    if len(x.shape) ==2:
        mean = x.mean(axis=0)
        variance = ((x - mean)**2).mean(axis=0)
    else:
        mean = x.mean(axis=(0, 2, 3), keepdim=True)
        variance = ((x - mean)**2).mean(axis=(0, 2, 3), keepdim=True)
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


class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        with self.name_scope:
            # path 1
            self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')
            # path 2
            self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
            self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1,  activation='relu')
            # path 3
            self.p3_conv_1 = nn.Conv2D(n3_1, kernel_size=1, activation='relu')
            self.p3_conv_5 = nn.Conv2D(n3_5, kernel_size=5, padding=2, activation='relu')
            # path 4
            self.p4_pool_3 = nn.MaxPool2D(pool_size=3, padding=1, strides=1)
            self.p4_conv_1 = nn.Conv2D(n4_1, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_conv_1(x)
        p2 = self.p2_conv_3(self.p2_conv_1(x))
        p3 = self.p3_conv_5(self.p3_conv_1(x))
        p4 = self.p4_conv_1(self.p4_pool_3(x))
        return nd.concat(p1, p2, p3, p4, dim=1)


class GoogLeNet(nn.Block):
    def __init__(self, num_classes, verbose=False, **kwargs):
        super(GoogLeNet, self).__init__(**kwargs)
        self.verbose = verbose
        with self.name_scope():
            # block 1
            b1 = nn.Sequential()
            b1.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                   nn.MaxPool2D(pool_size=3, strides=2)
                   )
            # block 2
            b2 = nn.Sequential()
            b2.add(nn.Conv2D(64, kernel_size=1),
                   nn.Conv2D(192, kernel_size=3, padding=1),
                   nn.MaxPool2D(pool_size=3, strides=2)
                   )

            # block 3
            b3 = nn.Sequential()
            b3.add(Inception(64, 192, 128, 16, 32, 32),
                   Inception(218, 128, 192, 32, 96, 64),
                   nn.MaxPool2D(pool_size=3, strides=2)
                   )
            # block 4
            b4 = nn.Sequential()
            b4.add(Inception(192, 96, 208, 16, 48, 64),
                   Inception(160, 112, 224, 24, 64, 64),
                   Inception(128, 128, 256, 24, 64, 64),
                   Inception(112, 144, 288, 32, 64, 64),
                   Inception(256, 160, 320, 32, 128, 128),
                   nn.MaxPool2D(pool_size=3, strides=2)
                   )

            # block 5
            b5 = nn.Sequential()
            b5.add(Inception(256, 160, 320, 32, 128, 128),
                   Inception(384, 192, 384, 48, 128, 128),
                   nn.AvgPool2D(pool_size=2)
                   )
            # block 6
            b6 = nn.Sequential()
            b6.add(nn.Flatten(),
                   nn.Dense(num_classes))

            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x,  *args):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output is: %s' %(i+1, out.shape))
        return out


class Residual(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(Residual, self).__init__()
        self.same_shape = same_shape
        with self.name_scope():
            strides = 1 if self.same_shape else 2
            self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1, strides=strides)
            self.bn1 = nn.BatchNorm()
            self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
            self.bn2 = nn.BatchNorm()
            if not same_shape:
                self.conv3 = nn.Conv2D(channels, kernel_size=1, strides=strides)

    def forward(self, x, *args):
        out = nd.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if not self.same_shape:
            x = self.conv3

        return nd.relu(out + x)


class ResNet18(nn.Block):
    def __init__(self, num_classe, verbose=False, **kwargs):
        self.verbose = verbose
        super(ResNet18, self).__init__()
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(channels=64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(nn.MaxPool2D(pool_size=3, strides=2),
                   Residual(channels=64),
                   Residual(channels=64)
                   )
            # block 3
            b3 = nn.Sequential()
            b3.add(Residual(channels=128, same_shape=False),
                   Residual(channels=128)
                   )

            # block 4
            b4 = nn.Sequential()
            b4.add(Residual(channels=256, same_shape=False),
                   Residual(channels=256)
                   )

            # block 5
            b5 = nn.Sequential()
            b5.add(Residual(channels=512, same_shape=False),
                   Residual(channels=512)
                   )

            # block 6
            b6 = nn.Sequential()
            b6.add(nn.AvgPool2D(pool_size=3),
                   nn.Dense(num_classe))

            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)
    def forward(self, x, *args):
        out = x
        for i, block in enumerate(self.net):
            out = block(net)
            if self.verbose:
                print('Block %d output: %s' %(i+1, out.shape))
        return out