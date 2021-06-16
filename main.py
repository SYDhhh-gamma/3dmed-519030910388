from functools import partial
import os
import config
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from tqdm import trange
from torch.utils.data import DataLoader
from dataset.fracnet_dataset import FracNetTrainDataset
from dataset import transforms as tsfm
from utils.metrics import dice, recall, precision, fbeta_score
from model.unet import UNet
from model.losses import MixLoss, DiceLoss



def val(model, val_loader, loss_func):
    model.eval()
    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.to(device), target.to(device)
            output = model(data)

            val_dice= dice(output, target)
    return val_dice.item()


def train(model, train_loader, optimizer, loss_func):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()


        train_dice= dice(output, target)

    return train_dice.item()


if __name__ == '__main__':
    args = config.args

    train_image_dir = args.train_image_dir
    train_label_dir = args.train_label_dir
    val_image_dir = args.val_image_dir
    val_label_dir = args.val_label_dir

    batch_size = 2
    num_workers = 4
    optimizer = optim.SGD
    criterion = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)

    thresh = 0.1
    recall_partial = partial(recall, thresh=thresh)
    precision_partial = partial(precision, thresh=thresh)
    fbeta_score_partial = partial(fbeta_score, thresh=thresh)

    model = UNet(1, 1, first_out_channels=16)
    model = nn.DataParallel(model.cuda())

    transforms = [
        tsfm.Window(-200, 1000),
        tsfm.MinMaxNorm(-200, 1000)
    ]
    ds_train = FracNetTrainDataset(train_image_dir, train_label_dir,
        transforms=transforms)
    train_loader = FracNetTrainDataset.get_dataloader(ds_train, batch_size, False,
        num_workers)
    ds_val = FracNetTrainDataset(val_image_dir, val_label_dir,
        transforms=transforms)
    val_loader = FracNetTrainDataset.get_dataloader(ds_val, batch_size, False,
        num_workers)


    save_path = os.path.join('./ouput', args.save)
    if not os.path.exists(save_path): os.makedirs(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # model info
    model =  UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    loss_func = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)
    best = [0, 0]  # 初始化最优模型的epoch和performance
    for epoch in trange(1, args.epochs + 1):
        train_log = train(model, train_loader, optimizer, loss_func)
        val_log = val(model, val_loader, loss_func)

        # Save checkpoint.
        state = {'net': model.state_dict(), 'dice':val_log, 'epoch': epoch}
        torch.save(state, os.path.join(save_path,  'ckpt_%d.pth' % (epoch)))
        if val_log > best[1]:
            best[0] = epoch
            best[1] = val_log
            trigger = 0

    print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))
