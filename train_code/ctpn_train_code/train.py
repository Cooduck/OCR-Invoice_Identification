#-*- coding:utf-8 -*-
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from ctpn.ctpn import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from ctpn.dataset import VOCDataset
from ctpn import config
import visdom
import matplotlib.pyplot as plt

random_seed = 2019
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

epochs = 80
lr = 1e-3
resume_epoch = 0


def save_checkpoint(state, epoch, loss_cls, loss_regr, loss, ext='pth'):
    check_path = os.path.join(config.checkpoints_dir, 'ctpn_ep{:02d}_{:.4f}_{:.4f}_{:.4f}.'.format(epoch, loss_cls, loss_regr, loss) + ext)
    try:
        torch.save(state, check_path)
    except BaseException as e:
        print(e)
        print('fail to save to {}'.format(check_path))
    print('saving to {}'.format(check_path))

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == '__main__':
    dataset = VOCDataset(config.img_dir, config.label_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_workers)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CTPN_Model().to(device)

    checkpoints_weight = config.pretrained_weights
    print('exist pretrained ',os.path.exists(checkpoints_weight))
    if os.path.exists(checkpoints_weight):
        print('using pretrained weight: {}'.format(checkpoints_weight))
        cc = torch.load(checkpoints_weight, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        resume_epoch = cc['epoch']
    else:
        model.apply(weights_init)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [35, 55, 70], gamma=0.1, last_epoch=-1)

    critetion_cls = RPN_CLS_Loss(device)
    critetion_regr = RPN_REGR_Loss(device)
    
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    best_model = None
    epochs += resume_epoch

    n_iter = 0
    for epoch in range(resume_epoch+1, epochs):
        print('Epoch {}/{}'.format(epoch, epochs))
        epoch_size = len(dataset) // 1
        model.train()
        epoch_loss_cls = 0
        epoch_loss_regr = 0
        epoch_loss = 0
        scheduler.step(epoch)
        for param_group in scheduler.optimizer.param_groups:
            print('lr: %s'% param_group['lr'])
        print('#'*80)

        Epoch_loss_cls = []
        Epoch_loss_regr = []
        Epoch_loss = []

        for batch_i, (imgs, clss, regrs) in enumerate(dataloader):
            since = time.time()
            imgs = imgs.to(device)
            clss = clss.to(device)
            regrs = regrs.to(device)
    
            optimizer.zero_grad()
    
            out_cls, out_regr = model(imgs)
            loss_regr = critetion_regr(out_regr, regrs)
            loss_cls = critetion_cls(out_cls, clss)
    
            loss = loss_cls + loss_regr 
            loss.backward()
            optimizer.step()
    
            epoch_loss_cls += loss_cls.item()
            epoch_loss_regr += loss_regr.item()
            epoch_loss += loss.item()
            mmp = batch_i + 1
            n_iter += 1
            print('time:{}'.format(time.time() - since))
            print(  'EPOCH:{}/{}--BATCH:{}/{}\n'.format(epoch, epochs-1, batch_i, epoch_size),
                    'batch: loss_cls:{:.4f}--loss_regr:{:.4f}--loss:{:.4f}\n'.format(loss_cls.item(), loss_regr.item(), loss.item()),
                    'epoch: loss_cls:{:.4f}--loss_regr:{:.4f}--loss:{:.4f}\n'.format(epoch_loss_cls/mmp, epoch_loss_regr/mmp, epoch_loss/mmp)
                )

            Epoch_loss_cls.append(epoch_loss_cls/mmp)
            Epoch_loss_regr.append(epoch_loss_regr/mmp)
            Epoch_loss.append(epoch_loss/mmp)

            if epoch % 1 == 0:
                iterations = range(epoch)  # 迭代次数

                # 创建一个新的图表
                plt.figure(figsize=(10, 5))

                # 绘制损失曲线
                plt.plot(iterations, Epoch_loss_cls,  label='Classification Loss')
                plt.plot(iterations, Epoch_loss_regr,  label='Regression Loss')
                plt.plot(iterations, Epoch_loss, label='Total Loss')

                # 添加图表标题和坐标轴标签
                plt.title('Loss Curves')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.legend()  # 显示图例

                if not os.path.exists('./log/'):
                    os.makedirs('./log/')

                # 保存图表到./log/目录下
                plt.savefig('./log/loss_curves.png')

        
        epoch_loss_cls /= epoch_size
        epoch_loss_regr /= epoch_size
        epoch_loss /= epoch_size
        print('Epoch:{}--{:.4f}--{:.4f}--{:.4f}'.format(epoch, epoch_loss_cls, epoch_loss_regr, epoch_loss))
        if best_loss_cls > epoch_loss_cls and best_loss_regr > epoch_loss_regr and best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            best_model = model
            save_checkpoint({'model_state_dict': best_model.state_dict(), 'epoch': epoch},
                            epoch,
                            best_loss_cls,
                            best_loss_regr,
                            best_loss)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
