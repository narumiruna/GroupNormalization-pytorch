
from torch.autograd import Variable
import torch.nn.functional as F


class Trainer(object):
    def __init__(self, model, optimizer, train_loader, val_loader, cuda=False, log_interval=10):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cuda = cuda
        self.log_interval = log_interval

        self.num_iters_per_epoch = len(self.train_loader.dataset) // self.train_loader.batch_size

    def train(self, epochs):
        for epoch in range(epochs):
            self.train_epoch(epoch)
            val_acc, val_avg_loss = self.validate()
            print('Validation acc: {:.2f}%, validation loss: {:.6f}.'.format(val_acc * 100, val_avg_loss))

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

            if (i + 1) % self.log_interval == 0:
                print('Train epoch: {},'.format(epoch + 1),
                      'iter: {}/{},'.format(i + 1, self.num_iters_per_epoch),
                      'train loss: {}.'.format(float(loss.data)))

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
        val_avg_loss = sum(loss_list) / len(self.val_loader.dataset)

        return val_acc, val_avg_loss
