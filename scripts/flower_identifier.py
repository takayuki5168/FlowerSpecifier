import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.training import extensions
from chainer.datasets import TransformDataset
from chainercv.transforms import resize
from chainer import cuda
import cupy as cp
from chainer import serializers


chainermodel = '/home/takayuki/.chainer/dataset/pfnet/chainer/models/VGG_ILSVRC_16_layers.npz'

class Model(chainer.Chain):
    #def __init__(self, out_size, chainermodel=chainermodel):
    def __init__(self, out_size):
        super(Model, self).__init__(
            #vgg = L.VGG16Layers(chainermodel),
            vgg = L.VGG16Layers(),
            fc1 = L.Linear(None, 1000),
            fc2 = L.Linear(None, 1000),
            fc3 = L.Linear(None, 1000),
            fc4 = L.Linear(None, 500)
        )

    def __call__(self, x, train=False):
        #with chainer.using_config('train', train):
        h = self.vgg(x, layers=['fc7'])['fc7']
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        y = self.fc4(h)

        return y

class FlowerIdentifier(object):
    def __init__(self):
        self.gpu_devices = {'main' : 0, 'second' : 1}
        #self.gpu_devices = {'main' : 0}
        chainer.backends.cuda.get_device_from_id(self.gpu_devices['main']).use()

    def transform(self, in_data):
        img, label = in_data
        img = resize(img, (224, 224))
        return img, label

    def train(self, epoch_num=40, batch_size=128, gpu=-1):
        train = chainer.datasets.LabeledImageDataset("../dataset/train/info.txt", "../dataset/train")
        test = chainer.datasets.LabeledImageDataset("../dataset/validation/info.txt", "../dataset/validation")

        model = L.Classifier(Model(out_size=25))   # loss function, default softmax_cross_entropy
        alpha = 1e-4
        optimizer = optimizers.Adam(alpha=alpha)
        optimizer.setup(model)
        model.predictor.vgg.disable_update()   # not update weight of VGG16

        train = TransformDataset(train, self.transform)
        test = TransformDataset(test, self.transform)

        train_iter = chainer.iterators.SerialIterator(train, batch_size)
        test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
        #updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
        updater = training.ParallelUpdater(train_iter, optimizer, devices=self.gpu_devices)
        trainer = training.Trainer(updater, (epoch_num, 'epoch'), out='result')
        trainer.extend(extensions.Evaluator(test_iter, model, device=self.gpu_devices['main']))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        #trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        #trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
        trainer.run()

        model.to_cpu()
        serializers.save_npz("mymodel.npz", model)


flower_identifier = FlowerIdentifier()
flower_identifier.train()
