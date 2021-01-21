import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.optimizer import GradualWarmupScheduler
from utils.dataset import load_data, TwoCropsTransform, augmentation_dictionary
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils.custom_typing import *
from utils.mypath import path_save, path_result, path_summary, path_ckpt

from tqdm import tqdm
import os


class BYOL():
    def __init__(self,
                 args,
                 encoder: torch.nn,  # encoder for online network / target network is copy of online network
                 predictor: torch.nn,  # predictor network comes after online network
                 optimizer: torch.optim,  # cosine annealing learning rate after warm-up, wo/ restart
                 **params):
        super(BYOL, self).__init__()
        self.args = args
        self.online_net = encoder.to(args.device)
        self.target_net = encoder.to(args.device)
        self.predictor = predictor.to(args.device)
        self.optimizer = optimizer
        self.max_epoch = args.max_epoch
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers  # num_workers for data loader
        self.device = args.device  # set cuda device
        self.writer = SummaryWriter(log_dir=path_summary)
        self.resume = args.resume


    @staticmethod
    def regression_loss(x, y):
        norm_x = F.normalize(x, dim=1)
        norm_y = F.normalize(y, dim=1)
        loss = 2-2*(norm_x*norm_y).sum(dim=-1)
        return loss

    @torch.no_grad()
    def _momentum_update_target_network(self, epoch):
        """
        Orginal From. MoCo v2  //  Customized to BYOL
        Momentum update of the target network linear update
        """
        # annealing tau from base_tau to 0.999
        tau = self.args.tau + (0.999-self.args.tau)*(epoch/self.max_epoch)
        for param_online, param_target in zip(self.online_net.parameters(), self.target_net.parameters()):
            param_target.data = (param_online.data * tau) + param_target.data*(1-tau)


    def train(self):
        # Parameters from online network stream
        params = list(self.online_net.parameters()) + list(self.predictor.parameters())

        # Set optimizer
        optimizer = self.optimizer(params, lr=self.args.learning_rate, momentum=0.9, weight_decay=0.0004)

        # Scheduling optimizer
        cosine_scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=100, eta_min=0, last_epoch=-1)

        # Warm-up wo/ restart
        scheduler = GradualWarmupScheduler(optimizer, multiplier=8, total_epoch=10, after_scheduler=cosine_scheduler)

        # Set data_loader (train)
        augmentation = augmentation_dictionary['covid']
        data_set = load_data(args=self.args,
                             split='train',
                             transform=TwoCropsTransform(base_transform=augmentation)).load_dataset()
        train_loader = DataLoader(data_set, batch_size=self.args.batch_size, shuffle=True,
                                  drop_last=False, num_workers=self.args.num_workers, pin_memory=True)

        def net_switch(view1: augmented_image, view2: augmented_image):
            # online network stream
            out_online = self.online_net(view1)
            out_online = self.predictor(out_online)

            # target network stream
            with torch.no_grad():
                out_target = self.target_net(view2)

            # get loss
            loss = self.regression_loss(out_online, out_target)

            # return loss
            return loss

        # set epoch number to start(+resume)
        start_epoch = 0
        if self.args.resume:
            checkpoint = torch.load(path_ckpt)
            start_epoch = checkpoint['epoch'] + 1
            print(f'Resume training ... EPOCH({start_epoch})')
            print(f'Get model from : {path_ckpt}')
            self.online_net.load_state_dict(checkpoint['online_network_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_network_state_dict'])
            self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loading Parameters Complete!')
        else:
            print("Start New Training!")

        # optimizer.zero_grad()
        # optimizer.step()
        for epoch in range(start_epoch, self.max_epoch):

            # Tensorboard loss per epoch
            epoch_loss = 0

            # pbar _ loss desc.
            loss = 0
            min_loss = 9999

            # steps for one epoch
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='loss_description')
            for i, (pos_pair, _) in pbar:
                # pbar _ dynamic desc
                pbar.set_description(f'| epoch: {epoch} | loss: {loss:.3f} | min_loss: {min_loss:.3f} | epoch_loss: {epoch_loss/(i+1):.3f} |')

                # get mini_batch with augmentation
                pos_pair[0] = pos_pair[0].to(self.args.device, non_blocking=True)
                pos_pair[1] = pos_pair[1].to(self.args.device, non_blocking=True)

                # get total losss
                loss1 = net_switch(pos_pair[0], pos_pair[1])
                loss2 = net_switch(pos_pair[1], pos_pair[0])
                loss = (loss1 + loss2).mean()

                epoch_loss += loss
                if loss < min_loss:
                    min_loss = loss

                # update parameters
                # scheduler.step()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.writer.add_scalar('loss', epoch_loss/(i+1), global_step=epoch)


            torch.save({
                'epoch': epoch + 1,
                'online_network_state_dict': self.online_net.state_dict(),
                'target_network_state_dict': self.target_net.state_dict(),
                'predictor_state_dict': self.predictor.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path_save+f'/byol_epoch_{epoch}_loss_{epoch_loss/(i+1):.2f}.ckpt')