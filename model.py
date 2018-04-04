
from torch import nn
from groupnorm import GroupNorm2d


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=None):
        super(ConvBlock, self).__init__()
        if norm == 'batch':
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )
        elif norm == 'group':
            self.main = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                GroupNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
            )
        else:
            raise ValueError('No such norm.')

    def forward(self, x):
        return self.main(x)


class Net(nn.Module):
    def __init__(self, norm, conv_ch=96):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            ConvBlock(3, conv_ch, 1, norm),
            ConvBlock(conv_ch, conv_ch, 1, norm),
            ConvBlock(conv_ch, conv_ch, 2, norm),

            ConvBlock(conv_ch, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 2, norm),

            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 2, norm),

            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 2, norm),

            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),
            ConvBlock(conv_ch * 2, conv_ch * 2, 1, norm),

            # output
            nn.Conv2d(conv_ch * 2, 10, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(x.size(0), -1)
        return out
