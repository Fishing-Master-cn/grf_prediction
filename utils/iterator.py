# -*- coding: utf-8 -*-
# @File  : iterator.py
# @Author: 汪畅
# @Time  : 2022/5/11  18:55
import random
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
from net.net import *
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


def train_one_epoch(model, device, train_loader, criterion, optimizer, idx, verbose=False, mixed=False):
    """
    在dataloader上完成一轮完整的迭代
    :param model: 网络模型
    :param device: cuda或cpu
    :param train_loader: 训练数据loader
    :param criterion: 损失函数
    :param optimizer: 优化器
    :param idx: 迭代轮数
    :param verbose: 是否打印进度条
    :return: training loss
    """
    model.train()
    print('\nEpoch {} starts, please wait...'.format(idx))

    # tqdm用于显示进度条
    loader = tqdm(train_loader)

    loss_list = []
    # 用于混合精度训练,可以加快运算速率
    scaler = GradScaler()

    for i, sample in enumerate(loader):
        train_data_batch = sample['X'].to(device).double()
        train_label_batch = sample['y'].to(device).double()

        if mixed:
            with autocast():  # 半精度加速训练
                output = model(train_data_batch)
                loss = criterion(output, train_label_batch.double())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        else:
            output = model(train_data_batch)
            loss = criterion(output, train_label_batch.double())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        loss_list.append(loss.item())
        loader.set_postfix_str(
            'lr:{:.8f}, loss: {:.6f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))
    torch.cuda.empty_cache()
    if not verbose:
        print('[ Training ] Lr:{:.8f}, Epoch Loss: {:.4f}'.format(optimizer.param_groups[0]['lr'], np.mean(loss_list)))
    return np.mean(loss_list)


# 调用torch.no_grad装饰器，验证阶段不进行梯度计算
@torch.no_grad()
def evaluate(model, device, test_loader, criterion):
    """
    模型评估
    :param model: 网络模型
    :param device: cuda或cpu
    :param test_loader: 测试数据loader
    :param criterion: 损失函数
    :param metric_list: 评估指标列表
    :return: test loss，评估指标
    """
    model.eval()  # 指定是模型evaluate而不是train,BN和DropOut不会取平均值

    prediction_list = []
    label_list = []
    mse_loss_list = []
    rmse_loss_list = []

    with torch.no_grad():

        for index, sample in enumerate(test_loader):
            test_data_batch = sample['X'].to(device).double()
            test_label_batch = sample['y'].to(device).double()

            prediction_data = model(test_data_batch).detach().cpu()
            test_label_batch = test_label_batch.detach().cpu()

            mse_loss = criterion(prediction_data, test_label_batch)
            rmse_loss = torch.sqrt(criterion(prediction_data, test_label_batch))
            mse_loss_list.append(mse_loss)
            rmse_loss_list.append(rmse_loss)

            prediction_list.append(np.array(prediction_data))
            label_list.append(np.array(test_label_batch))
            interval = 100

            if (index + 1) % interval == 0:
                prediction_list = np.array(prediction_list)
                label_list = np.array(label_list)
                r2 = r2_score(prediction_list[:, :, 0], label_list[:, :, 0], sample_weight=None,
                              multioutput='uniform_average')

                plt.figure(dpi=500)
                plt.grid(True)
                plt.plot(prediction_list[:, 0, 0], c='r')
                plt.plot(label_list[:, 0, 0])
                plt.legend(['Prediction', "True"])

                plt.savefig(r"test_result/mse_{:.4}_rmse_{:.4}_nrmse_{:.4}_r2_{:.4}.png".format(np.mean(
                    mse_loss_list),
                    np.mean(
                        rmse_loss_list),
                    np.mean(
                        rmse_loss_list) / (
                            np.max(
                                label_list) - np.min(
                        label_list)),
                    r2))
                plt.show()

                print('mse loss {} | rmse loss {} | nrmse loss {} | r2 {}'.format(

                    np.mean(
                        mse_loss_list),
                    np.mean(
                        rmse_loss_list),
                    np.mean(
                        rmse_loss_list) / (
                            np.max(
                                label_list) - np.min(
                        label_list)),
                    r2))

                prediction_list = []
                label_list = []
                mse_loss_list = []
                rmse_loss_list = []


def set_random_seed(seed=512, benchmark=True):
    """
    设定训练随机种子
    :param benchmark:
    :param seed: 随机种子
    :return: None
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    if not benchmark:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
