import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.training import extensions
from chainer.datasets import TransformDataset
from chainercv.transforms import resize

chainermodel = '/home/takayuki/.chainer/dataset/pfnet/chainer/models/VGG_ILSVRC_16_layers.npz'

class Model(chainer.Chain):
    #def __init__(self, out_size, chainermodel=chainermodel):
    def __init__(self, out_size):
        super(Model, self).__init__(
            #vgg = L.VGG16Layers(chainermodel),
            vgg = L.VGG16Layers(),
            fc = L.Linear(None, out_size)
        )

    def __call__(self, x, train=False):
        with chainer.using_config('train', train):
            h = self.vgg(x, layers=['fc7'])['fc7']
            y = self.fc(h)

        return y

class FlowerIdentifier(object):
    def __init__(self):
        pass

    def transform(self, in_data):
        img, label = in_data
        img = resize(img, (224, 224))
        return img, label

    def train(self, epoch_num=15, batch_size=128, gpu=-1):
        train = chainer.datasets.LabeledImageDataset("../dataset/train/info.txt", "../dataset/train")
        test = chainer.datasets.LabeledImageDataset("../dataset/validation/info.txt", "../dataset/validation")

        model = L.Classifier(Model(out_size=25))
        alpha = 1e-4
        optimizer = optimizers.Adam(alpha=alpha)
        optimizer.setup(model)

        train = TransformDataset(train, self.transform)
        test = TransformDataset(test, self.transform)

        train_iter = chainer.iterators.SerialIterator(train, batch_size)
        test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu)
        trainer = training.Trainer(updater, (epoch_num, 'epoch'), out='result')
        trainer.extend(extensions.Evaluator(test_iter, model, device=gpu))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))
        trainer.run()


flower_identifier = FlowerIdentifier()
flower_identifier.train()

exit()

class MyNet(chainer.Chain):
    def __init__(self, n_out):
        super(MyNet, self).__init__(
            base = L.VGG16Layers(),
            conv1 = L.Convolution2D(None, 10, 3, 3, 1),
            fc = L.Linear(None, n_out)
        )

    def __call__(self, x):
        #h = self.base(x)
        #h = self.fc(x)

        h = self.base(x, layers=["fc7"])
        h = self.fc(h["fc7"])

        #h = F.relu(self.conv1(x))
        #h = F.relu(self.fc(x))

        #h = F.relu(self.conv3(h))
        #h = F.relu(self.fc4(h))
        #h = self.fc5(h)

        return h

class FlowerIdentifier(chainer.Chain):
    def __init__(self):
        #super(FlowerIdentifier, self).__init__()

        self.load_dataset()
        self.init_trainer()

    def transform(self, in_data):
        img, label = in_data
        img = resize(img, (224, 224))
        return img, label

    def load_dataset(self):
        self.train_dataset = chainer.datasets.LabeledImageDataset("../dataset/train/info.txt", "../dataset/train")
        self.validation_dataset = chainer.datasets.LabeledImageDataset("../dataset/validation/info.txt", "../dataset/validation")

        self.train_dataset = TransformDataset(self.train_dataset, self.transform)
        self.validation_dataset = TransformDataset(self.validation_dataset, self.transform)

    def init_trainer(self, gpu_id=-1, batchsize=128, max_epoch=3):
        train_iter = iterators.SerialIterator(self.train_dataset, batchsize)
        validation_iter = iterators.SerialIterator(self.validation_dataset, batchsize, False)

        print("[Info] finish initializing iterator")
        net = L.Classifier(MyNet(n_out=25))
        #net.base.disable_update()
        #net = L.Classifier(net)
        optimizer = optimizers.SGD(lr=0.01).setup(net)


        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (max_epoch, "epoch"), out="logs")

        # add extension to trainer
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
        trainer.extend(extensions.Evaluator(validation_iter, net, device=gpu_id), name='val')
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
        trainer.extend(extensions.dump_graph('main/loss'))

        trainer.run()

    def __call__(self, x):
        pass

if __name__ == "__main__":
    flower_identifier = FlowerIdentifier()
    #flower_identifier.train()

