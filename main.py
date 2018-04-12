
import argparse
import os
from time import time

import torch
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms

from datasets import cifar10_loader
from models import CIFAR10Net
from trainers import Trainer

from bokeh import plotting


def run(config, norm2d):

    train_loader, valid_loader = cifar10_loader(config.root, config.batch_size)

    model = CIFAR10Net(norm2d=norm2d)
    if config.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    trainer = Trainer(model, optimizer, train_loader, valid_loader, use_cuda=config.cuda)

    valid_acc_list = []
    for epoch in range(config.epochs):
        start = time()

        scheduler.step()

        train_loss, train_acc = trainer.train(epoch)
        valid_loss, valid_acc = trainer.validate()

        print('epoch: {}/{},'.format(epoch + 1, config.epochs),
              'train loss: {:.4f}, train acc: {:.2f}%,'.format(train_loss, train_acc * 100),
              'valid loss: {:.4f}, valid acc: {:.2f}%,'.format(valid_loss, valid_acc * 100),
              'time: {:.2f}s'.format(time() - start))

        save_dir = os.path.join(config.save_dir, norm2d)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model_{:04d}.pt'.format(epoch + 1)))

        valid_acc_list.append(valid_acc)

    return valid_acc_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-dir', type=str, default='cifar10')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    group_valid_acc_list = run(args, 'group')
    batch_valid_acc_list = run(args, 'batch')

    p = plotting.figure(sizing_mode='stretch_both')
    x = range(len(group_valid_acc_list))
    p.line(x, group_valid_acc_list, line_color='green', alpha=0.5, line_width=5, legend='GroupNorm2d valid acc')
    p.line(x, batch_valid_acc_list, line_color='blue', alpha=0.5, line_width=5, legend='BatchNorm2d valid acc')

    os.makedirs(args.save_dir, exist_ok=True)
    f = os.path.join(args.save_dir, 'acc.html')
    plotting.output_file(f)
    plotting.save(p)


if __name__ == '__main__':
    main()
