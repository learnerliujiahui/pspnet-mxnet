import numpy as np
import random

import mxnet
import mxnet.ndarray as nd
import mxnet.autograd as ag
from mxnet import gluon


a = nd.zeros((3, 4), mxnet.gpu())
b = nd.ones((3, 4))

c = nd.array([[1, 2],[3, 4]])

x = nd.random_normal(0, 1, shape=(3, 4))


# convertion between numpy and mxnet

l = np.ones((2,1))
m = nd.array(l)  # numpy --> mxnet
n = m.asnumpy()  # mxnet --> numpy


# definr the data productor
def data_iter(num_data, batch_size=4):
    "
    para: num_data: length of the dataset
    para: batch_size: 
    "
    idx = list(range(num_data))
    random.shuffle(idx)
    for i in range(0, num_data, batch_size):
        batch = nd.array(idx[i:min(i + batch_ize, num_data)])
        yield nd.take(x, batch), nd.take(y, batch)

for data, label in data_iner():
    print() 

# initial the model parameterds
w = nd.random_normal(shape=(num_inputs, 1))
b = nd.zeros((1,))
params = [w, b]

for param in params:
    param.attach_grad()

# define the model

def net(x):
    return nd.dot(x, w) + b

def square_loss(yhat,y):
    return (yhat - y.reshape(yhat.shape)) ** 2

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

# start gluon module learning  

batcj_size = 10

dataset = gluon.data.ArrayDataset(x, y)
data_iter = gluon.data.Dataloader(dataset, batch_size, shuffle=True)

# define model
net = gluon.nn.Sequential()

# only define the output node number
net.add(gluon.nn.Dense(1))

# auto initialization
net.initialize()

# define loss
square_loss = gluon.loss.L2Loss()

# optimize
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})



epochs = 5
for e in range(epochs):
    total_loss = 0
    for data, label in data_iter:
        with ag.record():
            output = net(data)
            loss = square_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.sum(loss).asscalar()
    print("Epoch %d, average loss: %f" % (e, total_loss/num_data))




