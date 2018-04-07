
import argparse
from torchvision import datasets, transforms
from models import resnet
from torch import nn, optim
from trainer import Trainer
import torch
from torch.utils import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--norm', type=str, default='group', help='group | batch')
    parser.add_argument('--conv-ch', type=int, default=96)
    parser.add_argument('--parallel', action='store_true')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    parser.add_argument('--layers', nargs='+')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    transform = transforms.Compose([
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(360),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = data.DataLoader(datasets.CIFAR10(args.root,
                                                    train=True,
                                                    transform=transform,
                                                    download=True),
                                   batch_size=args.batch_size,
                                   shuffle=True)

    val_loader = data.DataLoader(datasets.CIFAR10(args.root,
                                                  train=False,
                                                  transform=test_transform,
                                                  download=True),
                                 batch_size=args.batch_size,
                                 shuffle=False)
    layers = list(map(int, args.layers))
    model = resnet.resnet_cifar(layers, norm=args.norm)
    print(model)
    if args.cuda:
        if args.parallel:
            model = nn.DataParallel(model)
        model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=5e-4, nesterov=True)

    #optimizer = optim.Adam(model.parameters(),
    #                       lr=args.lr,
    #                       betas=(args.beta1, args.beta2),
    #                       eps=args.eps,
    #                       weight_decay=1e-4)

    trainer = Trainer(model, optimizer, train_loader, val_loader, cuda=args.cuda)
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
