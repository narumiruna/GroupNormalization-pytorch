
import argparse
from torchvision import datasets, transforms
from model import Net
from torch import nn, optim
from trainer import Trainer
import torch
from torch.utils import data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--norm', type=str, default='group', help='group | batch')
    parser.add_argument('--conv-ch', type=int, default=96)
    parser.add_argument('--parallel', action='store_true')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_loader = data.DataLoader(datasets.CIFAR10(args.root,
                                                    train=True,
                                                    transform=transform,
                                                    download=True),
                                   batch_size=args.batch_size,
                                   shuffle=True)

    val_loader = data.DataLoader(datasets.CIFAR10(args.root,
                                                  train=False,
                                                  transform=transform,
                                                  download=True),
                                 batch_size=args.batch_size,
                                 shuffle=False)

    model = Net(norm=args.norm, conv_ch=96)
    if args.cuda:
        if args.parallel:
            model = nn.DataParallel(model)
        model.cuda()

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           betas=(args.beta1, args.beta2),
                           eps=args.eps)

    trainer = Trainer(model, optimizer, train_loader, val_loader, cuda=args.cuda)
    trainer.train(args.epochs)


if __name__ == '__main__':
    main()
