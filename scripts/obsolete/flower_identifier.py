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
    def __init__(self, out_size, chainermodel=chainermodel):
        super(Model, self).__init__(
            vgg = L.VGG16Layers(chainermodel),
            fc = L.Linear(None, out_size)
        )

    def __call__(self, x, train=True, extract_feature=False):
        with chainer.using_config('train', train):
            h = self.vgg(x, layers=['fc7'])['fc7']
            if extract_feature:
                return h
            y = self.fc(h)
        return y

gpu = -1

model = L.Classifier(Model(out_size=25))
alpha = 1e-4
optimizer = optimizers.Adam(alpha=alpha)
optimizer.setup(model)
model.predictor['fc'].W.update_rule.hyperparam.lr = alpha*10
model.predictor['fc'].b.update_rule.hyperparam.lr = alpha*10
#model.to_gpu(gpu)

epoch_num = 15
validate_size = 30
batch_size = 30



def transform(in_data):
    img, label = in_data
    img = resize(img, (224, 224))
    return img, label

train = chainer.datasets.LabeledImageDataset("../dataset/train/info.txt", "../dataset/train")
test = chainer.datasets.LabeledImageDataset("../dataset/validation/info.txt", "../dataset/validation")
train = TransformDataset(train, transform)
test = TransformDataset(test, transform)

#train, test = chainer.datasets.split_dataset_random(dataset, N-validate_size)
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




exit()


class MyNet(chainer.Chain):
    #def __init__(self, out_size, chainermodel=chainermodel):
    def __init__(self, out_size):
        super(MyNet, self).__init__(
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
        # init dataset
        train = chainer.datasets.LabeledImageDataset("../dataset/train/info.txt", "../dataset/train")
        test = chainer.datasets.LabeledImageDataset("../dataset/validation/info.txt", "../dataset/validation")

        # init network
        net = L.Classifier(MyNet(out_size=25))

        # init optimizer
        alpha = 1e-4
        optimizer = optimizers.Adam(alpha=alpha)
        optimizer.setup(net)

        # init iterator
        train = TransformDataset(train, self.transform)
        test = TransformDataset(test, self.transform)
        train_iter = chainer.iterators.SerialIterator(train, batch_size)
        test_iter = chainer.iterators.SerialIterator(test, batch_size, repeat=False, shuffle=False)

        # init updater
        updater = training.StandardUpdater(train_iter, optimizer, device=gpu)

        # init trainer
        trainer = training.Trainer(updater, (epoch_num, 'epoch'), out='log')
        trainer.extend(extensions.Evaluator(test_iter, net, device=gpu))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'], 'epoch', file_name='loss.png'))
        trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'], 'epoch', file_name='accuracy.png'))

        # train
        trainer.run()

flower_identifier = FlowerIdentifier()
flower_identifier.train()
