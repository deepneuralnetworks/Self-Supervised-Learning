import os
import numpy as np
import random

import torch
from torch.backends import cudnn
from torch.optim import SGD

from network import mlp_head, resnet
from utils.args import BYOL_parser_COVID
from utils.dataset import load_data, augmentation_dictionary
from utils.optimizer import LARS
from network.resnet import ResNet
from network.mlp_head import MLP_Head
from trainer import BYOL
from utils.mypath import path_save, path_result, path_summary


def main():
    # import arguments for experiment
    parser = BYOL_parser_COVID()
    args = parser.parse_args()

    # Hold randomness for reproducible experiment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(args.seed)  # numpy seed affects torchvision.transforms
    random.seed(args.seed)  # random seed affects data loader's sampling
    device = args.device
    print(f"Training with: {device}")

    # set encoder for online network & target network
    encoder = ResNet(args)

    # load pre-trained model if defined :TODO

    # set predictor network
    predictor = MLP_Head(in_channels=args.mlp_projection_size,
                         mlp_hidden_size=args.mlp_hidden_size,
                         projection_size=args.mlp_projection_size)

    # set optimizer
    optimizer = SGD

    # set learner
    trainer = BYOL(args, encoder, predictor, optimizer)

    # train learner
    trainer.train()

if __name__ == '__main__':
    os.makedirs(path_save, exist_ok=True)
    os.makedirs(path_result, exist_ok=True)
    os.makedirs(path_summary, exist_ok=True)

    main()