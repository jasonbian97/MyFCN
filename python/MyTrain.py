import sys
import os
from optparse import OptionParser
import argparse
import numpy as np
import time
from torch.optim import lr_scheduler
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
# customized import
# from [filename] import [function name]
from python.load import *
from python.utils import split_train_val, batch
from python.fcn import *
from python.metrics import *
###########################################################################
#                               Train                                     #
###########################################################################

def train_net(net,
              epochs=50,
              batch_size=10,
              lr=1e-4,
              val_percent=0.2,
              save_cp=False,
              gpu=True,
              dir_img="./",
              dir_mask="./",
              dir_checkpoint="./"):
    momentum   = 0
    w_decay    = 1e-5
    step_size  = 30
    gamma      = 0.3
    # return only the file name without extension.
    ids = get_ids(dir_img)
    # generate tuples like (id,#)
    ids = split_ids(ids)
    # split into train and val w.r.t val_percent
    iddataset = split_train_val(ids, val_percent)
    # show configeration info
    print('''
        Starting training:
            Epochs: {}
            Batch size: {}
            Learning rate: {}
            Training size: {}
            Validation size: {}
            Checkpoints: {}
            CUDA: {}
            step_size:{}
            gamma: {}
        '''.format(epochs, batch_size, lr, len(iddataset['train']),
                   len(iddataset['val']), str(save_cp), str(gpu),step_size,gamma))
    # use N_train to show where the training process is
    N_train = len(iddataset['train'])
    # set optimizer and criterion for loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(net.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    # start epoch loop
    for epoch in range(epochs):
        scheduler.step()
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        # set net mode to "train"
        net.train()
        # preprocess image in your dataset/dir, including:
        # 1. resize and crop
        # 2. transform image from HWC to CHW
        # 3. normalize images e.g. img/255.
        # 4. convert GT image into binary
        train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask)
        #
        epoch_loss = 0
        # start batch loop
        for i, b in enumerate(batch(train, batch_size)):
            # generate tensor batch
            imgs_list=[]
            mask_list=[]
            for k in b:
                imgs_list.append(k[0])
                mask_list.append(k[1])
            imgs = np.array(imgs_list).astype(np.float32)
            true_masks = np.array(mask_list).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            # The shape here is (Batchsize,C,W,H)
            # Load to training data to GPU
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()
            # inference once
            masks_pred = net(imgs)
            # flatten 4D matrix into 1D for computation simplicity
            masks_probs_flat = masks_pred.view(-1)
            true_masks_flat = true_masks.view(-1)
            # calculate loss and accumulate loss of each batch to calculate epoch loss
            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()
            # show where the training process is
            # print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))
            # clear the place for new grad
            optimizer.zero_grad()
            # BP
            loss.backward()
            # update optimizer to possibly change learning rate
            optimizer.step()
        # show loss os this Epoch
        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        # test model using val set
        (val_dice,val_iou) = eval_net(net, val, gpu)
        print('Validation Dice Coeff: {}; mIoU:{}'.format(val_dice,val_iou))
        # save model in dir_checkpoint
        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

###########################################################################
#                               Evaluate                                  #
###########################################################################

def eval_net(net, dataset, gpu=True):
    # set model to evaluation model
    net.eval()
    tot = 0
    tiou=0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]
        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)
        # The shape here is (1,W,H)
        if gpu:
            img = img.cuda()
            true_mask=true_mask.float().cuda()
        mask_pred = net(img)[0]
        mask_pred = (mask_pred > 0.5).float()

        # compute coefficient value
        tot += dice_coeff(mask_pred, true_mask).item()
        tiou += iou(mask_pred, true_mask,1)[0].item()

    return (tot / i,tiou/i)

###########################################################################
#                               get_args                                  #
###########################################################################
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=10,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('--n_class', dest='n_class', type='int',
                      default=1, help='n_class')
    # parser.add_option('-s', '--scale', dest='scale', type='float',
    #                   default=1., help='downscaling factor of the images')
    # Directory
    # parser.add_option('--dir_img', dest='dir_img', type='string',
    #                   default='/home/jasonbian/Desktop/WholeDataSet/DataSet1.0/cropped/')
    # parser.add_option('--dir_mask', dest='dir_mask', type='string',
    #                   default='/home/jasonbian/Desktop/WholeDataSet/DataSet1.0/BinaryMask3/')
    # parser.add_option('-dir_checkpoint', dest='dir_checkpoint', type='string',
    #                   default='./checkpoints/')

    (options, args) = parser.parse_args()
    return options

###########################################################################
#                               Main                                      #
###########################################################################
if __name__ == '__main__':

    # get global parameters: including: epochs,batchsize,lr,gpu
    args = get_args()
    # Directory
    dir_img="/home/jasonbian/Desktop/WholeDataSet/DataSet1.0/cropped/"
    dir_mask="/home/jasonbian/Desktop/WholeDataSet/DataSet1.0/BinaryMask3/"
    dir_checkpoint="./checkpoints/"
    # generate Model instance
    vgg_model = VGGNet(requires_grad=True, remove_fc=True)
    fcn_model = FCN8s(pretrained_net=vgg_model, n_class=args.n_class)
    # read global paprameters and do some preparation
    if args.load:
        fcn_model.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        ts = time.time()
        vgg_model = vgg_model.cuda()
        fcn_model = fcn_model.cuda()
        num_gpu = list(range(torch.cuda.device_count()))
        fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
        print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

    # read args and train Model
    try:
        train_net(net=fcn_model,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  dir_img=dir_img,
                  dir_mask=dir_mask,
                  dir_checkpoint=dir_checkpoint)
    #  save interrupt model
    except KeyboardInterrupt:
        torch.save(fcn_model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

