import mxnet
from mxnet import gluon
from mxnet import ndarray as nd
import matplotlib.pyplot as plt

def transform(data, label):
    return data.astype('float32')/255, label.astype('float32')

mnist_train = gluon.data.vision.FashionMNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.FashionMNIST(train=False, transform=transform)

data, label = mnist_train[0]

print('example shape: ', data.shape, 'label:', label)

def show_image(images):
    n = images.shape[0]
    _, figs = plt.subplots(1, n, figsize=(15, 15))
    for i in range(n):
        figs[i].imshow(images[i].reshape((28, 28)).asnumpy())
        figs[i].axes.get_xaxis().set_visible(False)
        figs[i].axes.get_yaxis().set_visible(False)
    plt.show()

def get_text_labels(label):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in label]

#data, label = mnist_train[0: 9]
#ahow_image(data)
#print(get_text_labels(label))


batch_size = 256
train_data = gluon.data.Dataloader(mnist_train, batch_size, shuffle=True)
test_data = gluon.data.Dataloader(mnist_test, batch_size, shuffle=True)

num_inputs = 784
num_outputs = 10

w = nd.random_normal(shape=(num_inputs, num_outputs))
b = nd.random_normal(shape=num_outputs)

params = [w, b]


for param in params:
    params.attach_grad()


def softmax(x):
    exp = nd.exp(x)
    pariition = exp.sum(axis=1, keepdim=True)
    return exp / partition

x =nd.random_normal(shape=(2, 5))
x_prob = softmax(x)

