
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, normalization):
        super(ConvBlock, self).__init__()
        if normalization:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                normalization(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )

    def forward(self, x):
        return self.main(x)


class Net(nn.Module):
    def __init__(self, normalization):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(3, 96, 1, normalization),
            ConvBlock(96, 96, 1, normalization),
            ConvBlock(96, 96, 2, normalization),

            ConvBlock(96, 192, 1, normalization),
            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 192, 2, normalization),

            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 192, 2, normalization),

            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 192, 2, normalization),

            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 192, 1, normalization),
            ConvBlock(192, 10, 2, None),
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(x.size(0), -1)
        return out


def test():
    import torch
    from torch.autograd import Variable
    from groupnorm import GroupNorm2d
    x = Variable(torch.randn(50, 3, 32, 32), volatile=True)
    net = Net(GroupNorm2d)
    print(net(x))


if __name__ == '__main__':
    test()
