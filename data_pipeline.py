# -*- coding: utf-8 -*-
# @File  : data_pipeline.py
# @Author: 汪畅
# @Time  : 2022/4/21  10:05

import os
import numpy as np
from torch.utils.data import Dataset
import xlrd


class FORCE_Dataset(Dataset):
    def __init__(self, data_dir, label_dir, subject_list, transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.subject_list = subject_list
        self.transform = transform
        self.file_list = self.get_file_list()

    def get_file_list(self):
        file_list = ['slice_' + i + '.xls' for i in self.subject_list]
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, i):
        data_filename = os.path.join(self.data_dir, self.file_list[i])
        label_filename = os.path.join(self.label_dir, self.file_list[i])

        X, y = data_loader(data_filename,
                           label_filename,
                           )

        sample = {'X': X, 'y': y}
        if self.transform:
            sample = self.transform(sample)

        return sample


# 文件读取主程序
def data_loader(X_root, y_root):
    """
    :param X_root: 输入特征的路径
    :param y_root: 标签的路径
    :return: 输入 输出特征
    """
    workbook_input = xlrd.open_workbook(X_root)
    worksheet_input = workbook_input.sheet_by_index(0)
    X_data = worksheet_input.row_values(0)
    X_data = np.array(X_data)

    workbook_output = xlrd.open_workbook(y_root)
    worksheet_output = workbook_output.sheet_by_index(0)
    y_data = worksheet_output.row_values(0)
    y_data = [y_data[1]]
    y_data = np.array(y_data)

    return X_data, y_data
