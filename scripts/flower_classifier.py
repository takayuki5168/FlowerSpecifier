from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models, utils
from torch.autograd import Variable
from IPython import embed
import numpy as np

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.network = models.vgg11(pretrained=True)

        # fix parameters of model.features
        for p in self.network.features.parameters():
            p.requires_grad = False

        self.network.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(4096, 25)
            )
        # self.network.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(4096, 100)
        #     )

    def forward(self, x):
        x = self.network(x)
        return x

class FlowerClassifier(object):
    def __init__(self, use_gpu=False, batch_size=256):
        self.use_gpu = use_gpu

        # init trainloader, testloader
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.train_data = datasets.ImageFolder(root='../dataset/train', transform=data_transform)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=batch_size, shuffle=True, num_workers=4)
        self.valid_data = datasets.ImageFolder(root='../dataset/validation', transform=data_transform)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=batch_size, shuffle=True, num_workers=4)

        # init model
        self.model = Model()
        if self.use_gpu:
            self.model = self.model.cuda()
            #self.model = torch.nn.DataParallel(self.model.cuda())

        # init loss function
        self.criterion = nn.CrossEntropyLoss().cuda() if self.use_gpu else nn.CrossEntropyLoss()

        # init optimizer
        self.optimizer = optim.Adam(self.model.network.classifier.parameters(), lr=0.01, weight_decay=1e-4)
        #self.optimizer = optim.Adam(self.model.parameters(), lr=0.01, weight_decay=1e-4)

        # init flower name
        self.flower_name_list = ['azalea', 'camellia', 'cluster_amaryllis', 'cosmos', 'daisy', 'dandelion', 'erigeron_philadelphicus', 'fleabane', 'houttuynia_cordata', 'hydrangea', 'iris', 'jasmine_flower', 'lantana_flower', 'marigold', 'mirabilis_jalapa', 'morning_glory', 'oxalis_corniculata_flower', 'pansy', 'rose', 'salvia_flower', 'sunflower', 'sweat_pea_flower', 'trifolium_pratense_flower', 'trifolium_repens', 'tulip']

    def train(self, epochs=10):
        print_step = 3
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            train_acc = 0
            for i, (inputs, labels) in enumerate(self.train_data_loader):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward, backward, optimize
                outputs = self.model(inputs.cuda().float()) if self.use_gpu else self.model(inputs)
                loss = self.criterion(outputs.cuda().float(), labels.cuda().long()) if self.use_gpu else self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                acc = (outputs.cuda().float().max(1)[1] == labels.cuda().long()).sum()
                train_acc += acc

            avg_train_loss = 1.0 * train_loss / len(self.train_data_loader.dataset)   # why we have to use 1.0 * 
            avg_train_acc = 1.0 * train_acc / len(self.train_data_loader.dataset)

            self.model.eval()
            # valid
            if True: #epoch % print_step == print_step - 1:
                with torch.no_grad():
                    valid_loss = 0
                    valid_acc = 0
                    for j, (inputs, labels) in enumerate(self.valid_data_loader):
                        #self.optimizer.zero_grad()
                        outputs = self.model(inputs.cuda().float()) if self.use_gpu else self.model(inputs)
                        loss = self.criterion(outputs.cuda(), labels.cuda().long()) if self.use_gpu else self.criterion(outputs, labels)
                        valid_loss += loss.item()
                        acc = (outputs.cuda().max(1)[1] == labels.cuda()).sum()
                        valid_acc += acc

                    avg_valid_loss = 1.0 * valid_loss / len(self.valid_data_loader.dataset)
                    avg_valid_acc = 1.0 * valid_acc / len(self.valid_data_loader.dataset)

            # print
            print('[%d, %5d] train_loss: %.3f valid_loss: %.3f, train_acc: %.3f valid_acc: %.3f' %
                  (epoch + 1, i + 1,
                   10000 * avg_train_loss,
                   10000 * avg_valid_loss,
                   100 * avg_train_acc,
                   100 * avg_valid_acc))

        print('Finished Trainig')

    def save_param(self, path):
        torch.save(self.model.state_dict(), path)

    def show_train_data(self):
        import matplotlib.pyplot as plt
        inputs, labels = iter(self.train_data_loader).next()
        img = utils.make_grid(inputs, nrow=8, padding=1)

        print([self.flower_name_list[int(i)] for i in labels])

        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.show()

    def show_valid_data(self):
        import matplotlib.pyplot as plt
        inputs, labels = iter(self.valid_data_loader).next()
        img = utils.make_grid(inputs, nrow=8, padding=1)

        print([self.flower_name_list[int(i)] for i in labels])

        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        plt.show()

    def load_param(self, path):
        self.model.load_state_dict(torch.load(path))

    def predict_train_data(self, show=False):
        with torch.no_grad():
            inputs, labels = iter(self.train_data_loader).next()
            outputs = self.model(inputs.cuda().float()) if self.use_gpu else self.model(inputs)
            acc = (outputs.cuda().float().max(1)[1] == labels.cuda().long()).sum()
            print('acc : {}'.format(1.0 * acc / len(inputs)))

    def predict_valid_data(self, show=False):
        with torch.no_grad():
            inputs, labels = iter(self.valid_data_loader).next()
            outputs = self.model(inputs.cuda().float()) if self.use_gpu else self.model(inputs)
            acc = (outputs.cuda().float().max(1)[1] == labels.cuda().long()).sum()
            print('acc : {}'.format(1.0 * acc / len(inputs)))
            if show:
                import matplotlib.pyplot as plt
                img = utils.make_grid(inputs, nrow=8, padding=1)

                print([[self.flower_name_list[int(labels[i])], self.flower_name_list[output.max(0)[1].item()]] for i, output in enumerate(outputs)])

                plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
                plt.show()


if __name__ == '__main__':
    if True:
        fc = FlowerClassifier(True)
        fc.load_param('./test_weight')
        fc.train(100)
        fc.save_param('./test_weight')
    else:
        fc = FlowerClassifier(True, 16)
        fc.load_param('./test_weight')
        fc.predict_train_data()
        fc.predict_valid_data(show=True)
