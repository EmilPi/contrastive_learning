# based on https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/packages/gluon/image/mnist.html
from __future__ import print_function

import mxnet as mx
import mxnet.ndarray as F
import numpy as np
from mxnet import autograd as ag
from mxnet import gluon
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

ctx = mx.gpu() if mx.test_utils.list_gpus() else mx.cpu()
EPS_ARR = F.array(np.array([1e-12])).as_in_context(ctx)


def train(net, trainer, train_data, epoch=100):
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    softmax_cross_entropy_loss = gluon.loss.SoftmaxCrossEntropyLoss()
    for i in range(epoch):
        train_data.reset()
        for batch in train_data:
            # Splits train data into multiple slices along batch_axis
            # and copy each slice into a context.
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=[ctx], batch_axis=0)
            # Splits train labels into multiple slices along batch_axis
            # and copy each slice into a context.
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=[ctx], batch_axis=0)
            outputs = []
            # Inside training scope
            with ag.record():
                for x, y in zip(data, label):
                    z, _ = net(x)
                    # Computes softmax cross entropy loss.
                    loss = softmax_cross_entropy_loss(z, y)
                    # Backpropagate the error for one iteration.
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
    return name, acc


# TODO https://github.com/apache/incubator-mxnet/discussions/20553
def augment_monochrome(x,
                       joint_transform,
                       noise_ampl=.2,
                       opencv_gpu_fix=True):
    print('augment_monochrome call', end='...')
    if opencv_gpu_fix:
        x_aug0 = x.as_in_context(mx.cpu())
    else:
        x_aug0 = x
    x_aug = F.repeat(x_aug0, repeats=3, axis=1)
    x_aug_ = F.stack(*[F.swapaxes(
        (1 + noise_ampl * np.random.normal()) * joint_transform(
            F.swapaxes(x_aug[i, ...], 0, 2)
        ) + noise_ampl * np.random.normal(), 0, 2
    ) for i in range(batch_size)])
    x_aug_ = F.broadcast_add(x_aug_, noise_ampl * F.random.normal(shape=x_aug_.shape))
    if opencv_gpu_fix:
        x_aug = x_aug_.as_in_context(ctx)
    else:
        x_aug = x_aug_
    x_aug = x_aug[:, :1, :, :]
    x_aug = F.clip(x_aug, 0., 1.)
    print('augment_monochrome finish', end='...')
    return x_aug


joint_transform = transforms.Compose([
    # transforms.RandomBrightness(.2),
    # transforms.RandomContrast(.3),
    transforms.RandomRotation(angle_limits=(-30, 30), zoom_in=True),
    transforms.RandomResizedCrop(size=28, scale=(.7, 1.), ratio=(.8, 1.25))
])


# based on incubator-mxnet/python/mxnet/gluon/loss.py CosineEmbeddingLoss class
def _cosine_similarity(x, y, axis=-1):
    x_norm = F.norm(x, axis=axis).reshape((-1, 1))
    y_norm = F.norm(y, axis=axis).reshape((-1, 1))
    x_dot_y = F.sum(x * y, axis=axis).reshape((-1, 1))
    EPS_ARR = F.array(np.array([1e-12])).as_in_context(ctx)
    return x_dot_y / F.broadcast_maximum(x_norm * y_norm, EPS_ARR)


def contrastive_pretrain(net, trainer, train_data, epoch=100):
    for i in range(epoch):
        cosine_similarity_loss = 0.
        print(f'contrastive_pretrain epoch {i}')
        train_data.reset()
        batch_count = 0
        for batch in train_data:
            batch_count += 1
            print(f'\rcontrastive pretraining batch {batch_count}', end='...')
            data = gluon.utils.split_and_load(batch.data[0], ctx_list=[ctx], batch_axis=0)
            label = gluon.utils.split_and_load(batch.label[0], ctx_list=[ctx], batch_axis=0)
            with ag.record():
                for x, y in zip(data, label):
                    _, z1 = net(x)
                    x_aug = augment_monochrome(x, joint_transform, noise_ampl=.2,
                                               opencv_gpu_fix=True,
                                               # opencv_gpu_fix=False
                                               )
                    x_aug = F.array(x_aug.asnumpy()).as_in_context(ctx)
                    _, z1aug = net(x_aug)
                    loss = 1. - _cosine_similarity(z1, z1aug)
                    loss.backward()
                    cosine_similarity_loss += loss.sum()
            trainer.step(batch.data[0].shape[0], ignore_stale_grad=True)
        print(f'epoch cosine similarity loss = {cosine_similarity_loss}')


def validate(net, val_data):
    # Use Accuracy as the evaluation metric.
    metric = mx.metric.Accuracy()
    val_data.reset()
    for batch in val_data:
        data = gluon.utils.split_and_load(batch.data[0], ctx_list=[ctx], batch_axis=0)
        label = gluon.utils.split_and_load(batch.label[0], ctx_list=[ctx], batch_axis=0)
        outputs = []
        for x in data:
            outputs.append(net(x)[0])
        metric.update(label, outputs)
    validation_metric = metric.get()
    return validation_metric


class Net(gluon.Block):
    def __init__(self, **kwargs):
        super(Net, self).__init__(**kwargs)
        with self.name_scope():
            self.conv1 = nn.Conv2D(4, kernel_size=(5, 5))
            self.pool1 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            self.conv2 = nn.Conv2D(4, kernel_size=(5, 5))
            self.pool2 = nn.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
            self.fc1 = nn.Dense(32)
            self.fc2 = nn.Dense(10)  # this is equal to number of classes

    def forward(self, x):
        x = self.pool1(F.tanh(self.conv1(x)))
        x = self.pool2(F.tanh(self.conv2(x)))
        x = x.reshape((0, -1))
        x_last = F.tanh(self.fc1(x))
        out = F.tanh(self.fc2(x_last))
        return out, x_last


def fix_seed(seed=33):
    mx.random.seed(seed, ctx=mx.cpu())
    mx.random.seed(seed, ctx=mx.gpu())
    np.random.seed(seed)


def init_net_and_data(batch_size):
    fix_seed()
    mnist = mx.test_utils.get_mnist()
    train_data = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_data = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
    fix_seed()
    net = Net()
    net.initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
    return train_data, val_data, net, trainer


def evaluate_final_network(net, val_data, training_metric=None):
    training_metric_name, training_metric_val = training_metric
    val_metric_name, val_metric_val = validate(net, val_data)
    print(f'validation {val_metric_name}={val_metric_val}')
    if training_metric_val and training_metric_name == val_metric_name:
        metric_overfit = training_metric_val - val_metric_val
        print(f'network is {"over" if metric_overfit > 0 else "under"}fitting:')
        print(f'\t{val_metric_name}_t - {val_metric_name}_v = {round(metric_overfit, 4)}')
    return val_metric_val


if __name__ == '__main__':
    batch_size = 100

    print('\n----------NORMAL TRAINING')
    train_data, val_data, net, trainer = init_net_and_data(batch_size)

    training_metric = train(net, trainer, train_data, epoch=32)

    validation_metric_val = evaluate_final_network(net, val_data, training_metric)

    #################################################################
    print('\n----------CONTRASTIVE PRETRAINING + NORMAL TRAINING!!!')
    train_data, val_data, net, trainer = init_net_and_data(batch_size)

    contrastive_pretrain(net, trainer, train_data, epoch=2)
    training_metric_contrastive_pretraining = train(net, trainer, train_data, epoch=32)

    validation_metric_contrastive_pretraining_val = evaluate_final_network(net, val_data,
                                                                           training_metric_contrastive_pretraining)

    if validation_metric_contrastive_pretraining_val > validation_metric_val:
        print('\n----------CONTRASTIVE PRETRAINING HELPED!')
    else:
        print('\n----------CONTRASTIVE PRETRAINING DID NOT HELP!')
