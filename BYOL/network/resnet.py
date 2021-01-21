import torchvision.models as models
import torch
import torch.nn as nn
from network.mlp_head import MLP_Head


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()

        self.args = args

        if args.arch == 'resnet18':
            resnet = models.resnet18(pretrained=False)
        elif args.arch == 'resnet50':
            resnet = models.resnet50(pretrained=False)
        else:
            raise ValueError()

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projector = MLP_Head(in_channels=resnet.fc.in_features,
                                  mlp_hidden_size=args.mlp_hidden_size,
                                  projection_size=args.mlp_projection_size)

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.shape[0], h.shape[1])
        return self.projector(h)