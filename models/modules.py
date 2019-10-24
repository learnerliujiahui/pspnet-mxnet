from mxnet.gluon import nn
from mxnet import autograd, gluon, nd, init

# AlexNet structure
class AlexNet(nn.Block):
    def __init__(self):
        super(AlexNet, self).__init__()
        with self.name_scope():
            self.net = nn.Sequential()
            # the first stage
            self.net.add(nn.Conv2D(channels=96, kernel_size=11, strides=4, activation='relu'))
            self.net.add(nn.MaxPool2D(pool_size=3, strides=2))
            # the second stage
            self.net.add(nn.Conv2D(channels=256, kernel_size=5, padding=2, activation='relu'))
            self.net.add(nn.MaxPool2D(pool_size=3, strides=2))
            # the third stage
            self.net.add(nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
            self.net.add(nn.Conv2D(channels=384, kernel_size=3, padding=1, activation='relu'))
            self.net.add(nn.Conv2D(channels=256, kernel_size=3, padding=1, activation='relu'))
            self.net.add(nn.MaxPool2D(pool_size=3, strides=2))
            # 4th stage
            self.net.add(nn.Flatten())
            self.net.add(nn.Dense(4096, activation='relu'))
            self.net.add(nn.Dropout(0.5))
            # 5th stage
            self.net.add(nn.Dense(4096, activation='relu'))
            self.net.add(nn.Dropout(0.5))
            # 6th stage
            self.net.add(nn.Dense(10))

    def forward(self, x, *args):
        out = self.net(x)
        return out


class VggNet(nn.Block):
    def __init__(self, conv_arch=((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))):
        super(VggNet, self).__init__()
        self.conv_arch = conv_arch
        with self.name_scope():
            self.net = nn.Sequential()

            for (num_convs, num_channels) in conv_arch:
                self.net.add(self.vgg_block(num_convs, num_channels))

            self.net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                    nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                    nn.Dense(10)
                    )

    # vgg block
    def vgg_block(num_convs, channels):
        out = nn.Sequential()
        with out.name_scope():
            for _ in range(num_convs):
                out.add(nn.Conv2D(channels=channels, kernel_size=3, padding=1, activation='relu'))
            out.add(nn.MaxPool2D(pool_size=2, strides=2))
        return out

    def forward(self, x, *args):
        out = self.net(x)
        return out


class Inception(nn.Block):
    def __init__(self, n1_1, n2_1, n2_3, n3_1, n3_5, n4_1, **kwargs):
        super(Inception, self).__init__(**kwargs)
        with self.name_scope:
            # path 1
            self.p1_conv_1 = nn.Conv2D(n1_1, kernel_size=1, activation='relu')
            # path 2
            self.p2_conv_1 = nn.Conv2D(n2_1, kernel_size=1, activation='relu')
            self.p2_conv_3 = nn.Conv2D(n2_3, kernel_size=3, padding=1, activation='relu')
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

    def forward(self, x, *args):
        out = x
        for i, b in enumerate(self.net):
            out = b(out)
            if self.verbose:
                print('Block %d output is: %s' % (i + 1, out.shape))
        return out


class ResidualBlock_v1(nn.Block):
    def __init__(self, channels, same_shape=True, **kwargs):
        super(ResidualBlock_v1, self).__init__()
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
    def __init__(self, num_classes, verbose=False, **kwargs):
        self.verbose = verbose
        super(ResNet18, self).__init__()
        with self.name_scope():
            # block 1
            b1 = nn.Conv2D(channels=64, kernel_size=7, strides=2)
            # block 2
            b2 = nn.Sequential()
            b2.add(nn.MaxPool2D(pool_size=3, strides=2),
                   ResidualBlock_v1(channels=64),
                   ResidualBlock_v1(channels=64)
                   )
            # block 3
            b3 = nn.Sequential()
            b3.add(ResidualBlock_v1(channels=128, same_shape=False),
                   ResidualBlock_v1(channels=128)
                   )

            # block 4
            b4 = nn.Sequential()
            b4.add(ResidualBlock_v1(channels=256, same_shape=False),
                   ResidualBlock_v1(channels=256)
                   )

            # block 5
            b5 = nn.Sequential()
            b5.add(ResidualBlock_v1(channels=512, same_shape=False),
                   ResidualBlock_v1(channels=512)
                   )

            # block 6
            b6 = nn.Sequential()
            b6.add(nn.AvgPool2D(pool_size=3),
                   nn.Dense(num_classes))

            self.net = nn.Sequential()
            self.net.add(b1, b2, b3, b4, b5, b6)

    def forward(self, x, *args):
        out = x
        for i, block in enumerate(self.net):
            out = block(out)
            if self.verbose:
                print('Block %d output: %s' % (i + 1, out.shape))
        return out


