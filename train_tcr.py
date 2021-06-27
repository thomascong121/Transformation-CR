'''
@author: Aamir Mustafa and Rafal K. Mantiuk
Implementation of the paper:
    Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation
    ECCV 2020

This file trains our method using only 20% of data as supervised data, rest is fed into the network in unsupervised fashion.
'''


import argparse
import hydra
import time
from math import log10
import pandas as pd
from tqdm.auto import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
from itertools import cycle
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Net
from data import get_training_set, get_test_set
import numpy as np
from pytorch_tcr import TCR   # Our file for generating the Transformation Matrix
from torch.utils.tensorboard import SummaryWriter

def hflip(input: torch.Tensor) -> torch.Tensor:
  return torch.flip(input, [-1])

tcr=TCR()  

weight= 0.01    

def train(cfg: DictConfig, epoch, writer):

    if cfg.Training.gpu_ids and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(cfg.Training.seed)

    device = torch.device("cuda" if cfg.Training.gpu_ids else "cpu")

    print('===> Loading datasets')
    train_set_target = get_training_set(cfg.Dataset.target_dataframe, cfg.Training.upscale_factor)
    train_set_source = get_training_set(cfg.Dataset.source_dataframe, cfg.Training.upscale_factor)
    test_set = get_test_set(cfg.Dataset.validation_dataframe, cfg.Training.upscale_factor)

    training_data_loader = DataLoader(dataset=train_set_target, num_workers=cfg.Training.n_worker, batch_size=cfg.Training.batchSize,
                                      shuffle=True)
    training_data_loader_un = DataLoader(dataset=train_set_source, num_workers=cfg.Training.n_worker, batch_size=cfg.Training.batchSize * 2,
                                      shuffle=True) # The batch size for unsupervised data is more than supervised data
    testing_data_loader = DataLoader(dataset=test_set, num_workers=cfg.Training.n_worker, batch_size=cfg.Training.batchSize,
                                     shuffle=False)

    print('===> Building model')
    model = Net(upscale_factor=opt.upscale_factor).to(device)
    criterion_mse = nn.MSELoss()
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    epoch_loss = 0

    with tqdm(enumerate(zip(cycle(training_data_loader), training_data_loader_un), 0), unit="batch", total=len(training_data_loader_un)) as tepoch:
        for iteration, batch in tepoch:
            tepoch.set_description(f"Epoch %d"%epoch)

            data_sup, data_un = batch[0], batch[1]
            input, target = data_sup[0].to(device), data_sup[1].to(device) # Here the data is used in supervised fashion
            input_un, target_un = data_un[0].to(device), data_un[1].to(device)  # Here the labels are not used

            # Applying our TCR on the Unsupervised data
            bs=  input_un.shape[0]
            random=torch.rand((bs, 1))
            transformed_input= tcr(input_un,random)

            optimizer.zero_grad()
            # Calculating the Unsupervised Loss
            loss_ours = criterion(model(transformed_input), tcr(model(input_un), random))
            #        print('Our Loss is ', loss_ours)

            loss = criterion(model(input), target)
            total_loss = loss + weight * loss_ours

            #        print('MSE Loss is ', total_loss)
            epoch_loss += total_loss.item()
            total_loss.backward()
            optimizer.step()

            # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), total_loss.item()))
            tepoch.set_postfix(loss=total_loss.item())

        print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
        writer.add_scalar('train_loss', epoch_loss / len(training_data_loader), epoch)

    print('===> Testing model')
    avg_psnr = 0
    with torch.no_grad():
        for batch in testing_data_loader:
            input, target = batch[0].to(device), batch[1].to(device)

            prediction = model(input)
            mse = criterion_mse(prediction, target)
            psnr = 10 * log10(1 / mse.item())
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))

    models_out_folder = os.path.join(cfg.Training.checkpoints_dir,
                                     cfg.Training.name,
                                     cfg.Training.subcat, 'Model')
    if not os.path.exists(models_out_folder):
        os.makedirs(models_out_folder)

    model_out_path = models_out_folder + "/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))



def save_images(cfg: DictConfig, epoch):

    model = torch.load('models/TCR/model_epoch_%d.pth'%epoch)
    model = model.cuda()
        
    
    test_path= cfg.Dataset.testing1_dataframe
    test_images= pd.read_csv(test_path)

    images_out_folder = os.path.join(cfg.Training.checkpoints_dir,
                                     cfg.Training.name,
                                     cfg.Training.subcat, 'Images')
    max_observation = cfg.Training.max_valid_dataset_size

    for index, row in test_images.iterrows():
        input_image = row.iloc[0]
        img = Image.open(input_image).convert('YCbCr')
        y, cb, cr = img.split()
    
    
        img_to_tensor = ToTensor()
        input_ = img_to_tensor(y).view(1, -1, y.size[1], y.size[0])
        input_ = input_.cuda()
    
        out = model(input_)
        out = out.cpu()
        out_img_y = out[0].detach().numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
    
        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

        if not os.path.exists(images_out_folder):
            os.makedirs(images_out_folder)
        out_img.save(images_out_folder +'/' + input_image)

        if index > max_observation:
            break
        
    print('output images saved')

@hydra.main(config_path='./configs', config_name='config.yaml')
def runner(cfg: DictConfig):
    log_out_folder = os.path.join(cfg.Training.checkpoints_dir,
                                     cfg.Training.name,
                                     cfg.Training.subcat, 'Logs')
    if not os.path.exists(log_out_folder):
        os.makedirs(log_out_folder)
    writer = SummaryWriter(log_out_folder)

    for epoch in range(1, cfg.Training.n_epoch + 1):
        train(cfg, epoch, writer)
        save_images(cfg, epoch)
    writer.close()



    
    






