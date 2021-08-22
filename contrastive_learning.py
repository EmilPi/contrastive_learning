# based on https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/packages/gluon/image/mnist.html
from __future__ import print_function
import mxnet as mx

from mxnet import gluon
from mxnet.gluon import nn
from mxnet import autograd as ag

import mxnet.ndarray as F


def train(net, train_data, epoch=100):
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
            outputs = []
            # Inside training scope
            with ag.record():
                for x, y in zip(data, label):
                    z, _ = net(x)
                    # Computes softmax cross entropy loss.
                    loss = softmax_cross_entropy_loss(z, y)
                    # Backpropogate the error for one iteration.
                    loss.backward()
                    outputs.append(z)
            # Updates internal evaluation
            metric.update(label, outputs)
            # Make one step of parameter update. Trainer needs to know the
            # batch size of data to normalize the gradient by 1/batch_size.
            trainer.step(batch.data[0].shape[0])
        # Gets the evaluation result.
        name, acc = metric.get()
        # Reset evaluation result to initial state.
        metric.reset()
        print('training acc at epoch %d: %s=%f' % (i, name, acc))


def validate(net, val_data):
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        # Splits validation data into multiple slices along batch_axis
        # and copy each slice into a context.
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=ctx, batch_axis=0)
        # Splits validation label into multiple slices along batch_axis
        # and copy each slice into a context.
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=ctx, batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x)[0])
        # Updates internal evaluation
        metric.update(label, outputs)
    validation_metric = metric.get()
    return validation_metric


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.conv1 = nn.Conv2D(2, kernel_size=(5, 5))
            self.pool1 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            self.conv2 = nn.Conv2D(2, kernel_size=(5, 5))
            self.pool2 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            self.fc1 = nn.Dense(16)
            self.fc2 = nn.Dense(10)  # this is equal to number of classes

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        # 0 means copy over size from corresponding dimension.
        # -1 means infer size from the rest of dimensions.
        x = x.reshape((0, -1))
        x_last = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(x_last))
        return out, x_last


if __name__ == '__main__':
    # Fixing the random seed
    mx.random.seed(33)

    mnist = mx.test_utils.get_mnist()

    batch_size = 100
    train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    net = Net()

    # set the context on GPU is available otherwise CPU
    gpus = mx.test_utils.list_gpus()
    ctx = [mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()]
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

    train(net, train_data)

    validation_metric = validate(net, val_data)
    print('validation acc: %s=%f' % validation_metric)
    # assert validation_metric[1] > 0.98 # only if network is good
