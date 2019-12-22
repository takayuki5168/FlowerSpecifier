import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.network = models.vgg11(pretrained=True)

        # fix parameters of model.features
        for p in self.network.features.parameters():
            p.requires_grad = False

        self.network.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
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
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu

        # init trainloader, testloader
        data_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()])
        self.train_data = datasets.ImageFolder(root='../dataset/train', transform=data_transform)
        self.train_data_loader = torch.utils.data.DataLoader(self.train_data, batch_size=64, shuffle=True, num_workers=4)
        self.valid_data = datasets.ImageFolder(root='../dataset/validation', transform=data_transform)
        self.valid_data_loader = torch.utils.data.DataLoader(self.valid_data, batch_size=64, shuffle=False, num_workers=4)

        # init model
        self.model = Model()
        if self.use_gpu:
            self.model = self.model.cuda()

        # init loss function
        self.criterion = nn.CrossEntropyLoss().cuda() if self.use_gpu else nn.CrossEntropyLoss()

        # init optimizer
        self.optimizer = optim.Adam(self.model.network.classifier.parameters(), lr=0.01, weight_decay=1e-4)


    def train(self, epochs=10):
        self.model.train()

        print_step = 3
        for epoch in range(epochs):
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
            avg_train_loss = train_loss / len(self.train_data_loader.dataset)
            avg_train_acc = train_acc / len(self.train_data_loader.dataset)

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

                    avg_valid_loss = valid_loss / len(self.valid_data_loader.dataset)
                    avg_valid_acc = valid_acc / len(self.valid_data_loader.dataset)

            # print
            print('[%d, %5d] train_loss: %.3f valid_loss: %.3f, train_acc: %.3f valid_acc: %.3f' %
                  (epoch + 1, i + 1,
                   10000 * avg_train_loss,
                   10000 * avg_valid_loss,
                   10000 * avg_train_acc,
                   10000 * avg_valid_acc))

        print('Finished Trainig')

    def save_param(self, path):
        torch.save(self.model.state_dict(), path)

if __name__ == '__main__':
    fc = FlowerClassifier(True)
    fc.train(50)
