
import configparser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import yaml

from pathlib import Path
from torch.utils.data import DataLoader

from dataset import UnpairedDataset
from model import SIFA
from utils import parse_config, get_config, set_random

# train
def train():
    # load config
    config = "./config/train.cfg"
    config = parse_config(config)
    #print(config)
    # load data
    trainset = UnpairedDataset(config['train']['a_path'], config['train']['b_path'],
                               config['train']['random'])
    train_loader = DataLoader(trainset, config['train']['batch_size'],
                              shuffle=True, drop_last=True)
    # Various need miscellaneous
    exp_name = config['train']['exp_name']
    save_path = Path(config['train']['save_path']).joinpath(exp_name).expanduser()
    model_path = save_path.joinpath("model-weights")
    model_path.mkdir(parents=True, exist_ok=True)

    loss_cycle = []
    loss_seg = []
    # load model
    if "gpu" in config['train'].keys():
        device = torch.device(f"cuda:{config['train']['gpu']}")
    else:
        device = torch.device('cpu')
    sifa_model = SIFA(config).to(device)
    sifa_model.train()
    sifa_model.initialize()
    num_epochs = config['train']['num_epochs']
    save_epoch = config['train']['save_every_epoch']

    for epoch in range(num_epochs):
        for i, (A, A_label, B, _) in enumerate(train_loader):

            A = A.to(device).detach()
            B = B.to(device).detach()
            A_label = A_label.to(device).detach()

            sifa_model.update_GAN(A, B)
            sifa_model.update_seg(A, B, A_label)
            loss_cyclea, loss_cycleb, segloss = sifa_model.print_loss(epoch)
            loss_cycle.append(loss_cyclea + loss_cycleb)
            loss_seg.append(segloss)
        # ddfseg_model.update_lr() #no need for changing lr
        if (epoch + 1) % save_epoch == 0:
            sifa_model.sample_image(epoch, save_path)
            fname = model_path.joinpath(f"model-{epoch + 1:03d}.pth")
            torch.save(sifa_model.state_dict(), str(fname))
        sifa_model.update_lr()

    print('Train finished! Just getting you a nice summary now.')
    opath = save_path.joinpath("train-losses")
    opath.mkdir(parents=True, exist_ok=True)
    loss_cycle, loss_seg = np.array(loss_cycle), np.array(loss_seg)
    np.savez(opath.joinpath('trainingloss.npz'), loss_cycle, loss_seg)
    x = np.arange(0, loss_cycle.shape[0])
    plt.figure(1) #FIXME: Augment figure resolution
    plt.plot(x, loss_cycle, label="Training's cycle loss")
    plt.legend()
    plt.xlabel('Iterations') #FIXME: Also get loss per epoch
    plt.ylabel('Cycle loss')
    plt.savefig(opath.joinpath('cycleloss.jpg'))
    plt.close()
    plt.figure(2)
    plt.plot(x, loss_seg, label="Training's segmentation loss")
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Segmentation loss')
    plt.savefig(opath.joinpath('segloss.jpg'))
    plt.close()
    print('Losses saved!')


if __name__ == '__main__':
    set_random()
    train()
