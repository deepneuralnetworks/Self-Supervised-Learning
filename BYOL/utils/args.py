import torch
import argparse

from utils.custom_typing import *

def BYOL_parser_COVID():
    parser = argparse.ArgumentParser(description='default: BYOL covid-19')

    # BYOL option
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--mlp_hidden_size', type=int, default=512)
    parser.add_argument('--mlp_projection_size', type=int, default=128)



    # Dataset option
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', type=torch.device, default=torch.device('cuda:0'))


    # Training option
    parser.add_argument('--resume', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2020010553)
    parser.add_argument('--max_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.025,
                        help='learning_rate = 0.2*BatchSize/256')
    parser.add_argument('--warmup', type=float, default=10)
    parser.add_argument('--weight_decay', type=float, default=1.5*1e-6)
    parser.add_argument('--tau_base', type=float, default=0.999,
                        help="""For the target network, the EMA parameter tau starts from tau_base.
                             tau = 1-(1-tau_base)*(cos(\pi.k/K)+1)/2 
                             (k:current training step) (K:maximum number of training steps)""")


    return parser