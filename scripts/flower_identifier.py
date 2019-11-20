import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import training
from chainer import iterators
from chainer.training import extensions
from chainer.datasets import TransformDataset
from chainercv.transforms import resize

class MyNet(chainer.Chain):
    def __init__(self, n_out):
        super(MyNet, self).__init__(
            base = L.VGG16Layers(),
            fc = L.Linear(None, n_out)
        )

    def __call__(self, x):
        h = self.base(x, layers=["fc7"])
        h = self.fc(h["fc7"])
        #h = F.relu(self.conv3(h))
        #h = F.relu(self.fc4(h))
        #h = self.fc5(h)
        return h

class FlowerIdentifier(chainer.Chain):
    def __init__(self):
        super(FlowerIdentifier, self).__init__()

        self.load_dataset()
        self.init_trainer()

    def transform(self, in_data):
        img, label = in_data
        img = resize(img, (256, 256))
        return img, label

    def load_dataset(self):
        self.train_dataset = chainer.datasets.LabeledImageDataset("../dataset/train/info.txt", "../dataset/train")
        self.validation_dataset = chainer.datasets.LabeledImageDataset("../dataset/validation/info.txt", "../dataset/validation")

        self.train_dataset = TransformDataset(self.train_dataset, self.transform)
        self.validation_dataset = TransformDataset(self.validation_dataset, self.transform)

    def init_trainer(self, gpu_id=-1, batchsize=128, max_epoch=3):
        train_iter = iterators.SerialIterator(self.train_dataset, batchsize)
        validation_iter = iterators.SerialIterator(self.validation_dataset, batchsize, False)

        net = MyNet(n_out=25)
        net.base.disable_update()
        net = L.Classifier(net)
        optimizer = optimizers.SGD(lr=0.01).setup(net)


        updater = training.StandardUpdater(train_iter, optimizer, device=gpu_id)
        trainer = training.Trainer(updater, (max_epoch, "epoch"), out="po")

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

