
from torch.autograd import Variable
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, cuda=False):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cuda = cuda

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            self.validate()

    def train_epoch(self, epoch):
        for i, (x, y) in enumerate(self.train_loader):
            x = Variable(x)
            y = Variable(y)

            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            pred = self.model(x)
            loss = F.cross_entropy(pred, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate(self):
        correct = 0
        loss_list = []

        for i, (x, y) in enumerate(self.val_loader):
            x = Variable(x, volatile=True)
            y = Variable(y)

            if self.cuda:
                x = x.cuda()
                y = y.cuda()

            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss_list.append(float(loss.data) * x.size(0))

            y_pred = output.data.max(dim=1)[1]
            correct += int(y.data.eq(y_pred).cpu().sum())

        val_acc = correct / len(self.val_loader.dataset)
        avg_loss = sum(loss_list) / len(self.val_loader.dataset)

        return val_acc, avg_loss
