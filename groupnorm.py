import torch
from torch import nn
from torch.nn import Parameter


class GroupNorm2d(nn.Module):
    def __init__(self, num_features, num_groups=16, eps=1e-5, affine=True):
        super(GroupNorm2d, self).__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.num_features = num_features
        self.affine = affine

        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features, 1, 1))
            self.bias = Parameter(torch.Tensor(1, num_features, 1, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1)
            self.bias.data.zero_()

    def forward(self, input):
        output = input.view(input.size(0), self.num_groups, -1)

        mean = output.mean(dim=2, keepdim=True)
        var = output.var(dim=2, keepdim=True)

        output = (output - mean) / (var + self.eps).sqrt()
        output = output.view_as(input)

        if self.affine:
            output = output * self.weight + self.bias

        return output

    def __repr__(self):
        return '{name}({extra_repr})'.format(name=self.__class__.__name__, extra_repr=self.extra_repr())

    def extra_repr(self):
        return '{num_features}, num_groups={num_groups}, eps={eps}, affine={affine}.'.format(**self.__dict__)
