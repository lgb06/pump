from pathlib import Path
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.repository import get_dataset
import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import h5py
import bisect
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/dataWX/WYL/PHM-Large-Model')
# from data_provider.m4 import M4Dataset, M4Meta # removed due to 
import warnings
import torch.nn.functional as F

warnings.filterwarnings('ignore')

PHM_LABEL_TO_INDEX = {
    'B007': 0,
    'B014': 1,
    'B021': 2,
    'B028': 3,
    'IR007': 4,
    'IR014': 5,
    'IR021': 6,
    'IR028': 7,
    'OR007_3': 8,
    'OR007_6': 9,
    'OR007_12': 10,
    'OR014_6': 11,
    'OR021_3': 12,
    'OR021_6': 13,
    'OR021_12': 14,
    'normal': 15,
    'broken': 16,
    'wear': 17,
    'pitting': 18,
    'pitting_wear': 19,
    'chip': 20,
    'N': 21,
    'IR_1': 22,
    'IR_2': 23,
    'IR_3': 24,
    'IR_4': 25,
    'OR_1': 26,
    'OR_2': 27,
    'OR_3': 28,
    'OR_4': 29,
    'RO_1': 30,
    'RO_2': 31,
    'RO_3': 32,
    'RO_4': 33,
}

class Dataset_PHM(Dataset):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, flag=None, cross_condition=None,
                 down_sampling_scale=1, data_path=None, start_percentage=None, end_percentage=None, padd_channel_num =None
                ,few_shot = False):
        
        self.condition_code = {}
        self.channel_code = {}

        self.__root_path = root_path
        self.__data_path = data_path
        self.__file_path = Path(self.__root_path) / self.__data_path
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.flag = flag
        self.cross_condition = cross_condition
        self.feature_size = None
        self.length = None
        self.args = args
        self.file_list = []
        self.global_idxes = []
        self.local_idxes = []
        self.file_data = []
        self.file_label = []
        self.file_condition = []
        self.file_channel_names = []
        self.few_shot = few_shot

        self.__read_data__()
        if padd_channel_num is not None:
            self.feature_size[1] = padd_channel_num
            self.__pad_channel__()
        self.__standardize__()
        self.global_label_index = PHM_LABEL_TO_INDEX

    def __standardize__(self):
        # Standardize the data
        # 讲所有的数据进行标准化
        self.scaler = StandardScaler()
        x_data_all = []
        for i in range(len(self.file_data)):
            x_data_all.append(self.file_data[i])
        x_data_all = np.vstack(x_data_all)
        self.scaler.fit(x_data_all)
        for i in range(len(self.file_data)):
            self.file_data[i] = torch.tensor(self.scaler.transform(self.file_data[i]), dtype=torch.float32)
        return None
    def __pad_channel__(self):
        # 找到所有数据中的最大通道数
        max_channels = max(data.shape[1] for data in self.file_data)
        min_channels = min(data.shape[1] for data in self.file_data)
        if max_channels != min_channels:
            print(f'[{Path(self.__root_path).name}] The number of channels is not consistent, padding the channel')
        # 遍历每个数据文件的通道数
            for i, (data, channel_names) in enumerate(zip(self.file_data, self.file_channel_names)):
                current_channels = data.shape[1]
                # 若当前通道数小于最大通道数，进行补齐
                if current_channels < max_channels:
                    # 计算需要补齐的通道数
                    repeat_count = max_channels - current_channels
                    # 通过重复现有通道来补齐通道
                    new_data = np.concatenate([data] + [data[:, -1:]] * repeat_count, axis=1)
                    new_channel_names = np.append(channel_names, [channel_names[-1]] * repeat_count)
                    # 更新数据和通道名
                    self.file_data[i] = new_data
                    self.file_channel_names[i] = new_channel_names
        # 在填充通道后，检查所有数据的通道数是否一致
        channel_counts = set(data.shape[1] for data in self.file_data)
        if len(channel_counts) == 1:
            print(f'所有数据的通道数已统一，为 {channel_counts.pop()}')
        else:
            print(f'数据的通道数仍然不一致，通道数集合为 {channel_counts}')
        return None



    def __read_data__(self):
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.file_list = [i for i in self.__file]

        self.feature_size = [self.seq_len, self.__file[self.file_list[0]][:].shape[1]]

        if self.cross_condition is not None:
            # cross_condition 不为None时，表示按工况划分训练集和测试集
            # cross_condition 应为一个长度为工况数的 [0, 1] 列表，表明在 train 的时候是否使用该工况，1表示使用，0表示不使用
            # 在 test 的时候列表值要取反
            print(f'Cross condition setting when [{self.flag} stage]')
            
            condition_set = set([self.__file[file].attrs['condition'] for file in self.file_list])

            # 检查该数据集是否存在多个工况
            if len(condition_set) > 1:
                # 如果存在，检查cross_condition的长度是否等于工况数
                assert len(condition_set) == len(self.cross_condition), 'The condition set is not equal to the cross_condition'

                # 按工况取出训练集或测试集
                self.file_list = self.__condition_masker__(condition_set)
                # self.start_percentage 和 self.end_percentage 不需要进行修改

            else:
                # 如果不存在多个工况，则无法按工况划分训练集和测试集，退出程序
                raise Exception('The dataset has only one condition, can not split the dataset by condition')

        else:
            # cross_condition为None时，表示不按工况划分训练集和测试集
            # 直接使用所有的数据按8：2切分
            # 非跨工况的数据集切分方式
            print(f'Non-cross condition setting when [{self.flag} stage]')
            
            if self.flag == 'pretrain':
                pass
            # 需要根据 train 和 test 修改 start_percentage 和 end_percentage
            elif self.flag == 'train':
                # if self.start_percentage is None:
                self.start_percentage = 0.0
                self.end_percentage = 0.8
            elif self.flag == 'test':
                self.start_percentage = 0.8
                self.end_percentage = 1.0
            print(f'[{Path(self.__root_path).name}] start_percentage: {self.start_percentage}, end_percentage: {self.end_percentage}')
        
        global_index = 0
        for file in self.file_list:
            data = self.__file[file][:]
            for i in range(self.down_sampling_scale):
                self.global_idxes.append(global_index)
                data_i = data[i::self.down_sampling_scale]
                file_length = data_i.shape[0]
                self.file_data.append(data_i)
                self.file_label.append(self.__file[file].attrs['label'])
                self.file_condition.append(self.__file[file].attrs['condition'])
                self.file_channel_names.append(self.__file[file].attrs['channel_name'])
                total_sample_num = (file_length - self.seq_len) // self.stride_len + 1
                start_idx = int(total_sample_num * self.start_percentage) * self.stride_len
                end_idx = int(total_sample_num * self.end_percentage-1) * self.stride_len
                self.local_idxes.append(start_idx)
                global_index += (end_idx - start_idx) // self.stride_len + 1
        self.length = global_index
        # Create a mapping from condition names to unique integers
        self.condition_code = {condition: idx for idx, condition in enumerate(sorted(set(self.file_condition)))}
        if self.few_shot and self.flag =='train':
            self.__process_few_shot__()
        return None

    def __process_few_shot__(self):        
        # 读取所有的label和file
        print(f'Few-shot setting when [{self.flag} stage]')
        label_to_files = {}
        for file, label in zip(self.file_list, self.file_label[::self.down_sampling_scale]):
            if label not in label_to_files:
                label_to_files[label] = []
            label_to_files[label].append(file)

        #读取每个label里的最小文件数
        min_file_num = min([len(files) for files in label_to_files.values()])
        shot_file_num = min(min_file_num, self.args.shot_num)
        print(f'Few-shot setting: shot_num: {self.args.shot_num}, shot_file_num: {shot_file_num}')
        # 按照每个label随机选择shot个文件
        few_shot_files = []
        for label, files in label_to_files.items():
            # if shot_file_num == 1:
            #     few_shot_files += [files[0]]
            # else:
            few_shot_files += np.random.choice(files, shot_file_num, replace=False).tolist()

        # 更新file_list和其他相关属性
        self.file_list = few_shot_files
        self.global_idxes = []
        self.local_idxes = []
        self.file_data = []
        self.file_label = []
        self.file_condition = []
        self.file_channel_names = []

        global_index = 0
        for file in self.file_list:
            data = self.__file[file][:]
            self.global_idxes.append(global_index)
            data_i = data[::self.down_sampling_scale]
            file_length = data_i.shape[0]
            self.file_data.append(data_i)
            self.file_label.append(self.__file[file].attrs['label'])
            self.file_condition.append(self.__file[file].attrs['condition'])
            self.file_channel_names.append(self.__file[file].attrs['channel_name'])
            if self.flag == 'train':
                total_sample_num = min(
                    (file_length - self.seq_len) // self.stride_len + 1,
                    self.args.shot_num//shot_file_num)
            else:
                total_sample_num = (file_length - self.seq_len) // self.stride_len + 1
            start_idx = int(total_sample_num * self.start_percentage) * self.stride_len
            end_idx = int(total_sample_num * self.end_percentage - 1) * self.stride_len
            self.local_idxes.append(start_idx)
            global_index += (end_idx - start_idx) // self.stride_len + 1

        self.length = global_index




    def __condition_masker__(self, condition_set):

        if self.flag == 'pretrain':
            print('Use all condition')
            return self.file_list
        elif self.flag == 'train':
            pass
        elif self.flag == 'test':
            self.cross_condition = [1 - x for x in self.cross_condition]
        
        used_condition_set = [condition for condition, mask in zip(condition_set, self.cross_condition) if mask]
        
        print(f'[{Path(self.__root_path).name}] All condition: {condition_set}')
        print(f'[{Path(self.__root_path).name}] Used condition: {used_condition_set}')

        new_file_list = []
        
        for file in self.file_list:
            if self.__file[file].attrs['condition'] in used_condition_set:
                new_file_list.append(file)
        
        return new_file_list


    
    def show_file_info(self):
        print(f"Channel_names_list: {self.__file[self.file_list[0]].attrs['channel_name']}")
        point_number = 0
        for i, file in enumerate(self.file_list):
            data = self.__file[file][:]
            point_number += data.shape[0]
            attrs = self.__file[file].attrs
            attrs_str = ' \t| '.join([f"No. {key}: {len(val)}" if key in ['channel_names', 'channel_name'] else f"{key}: {val}" for key, val in attrs.items()])
            print(f"File {(i+1):02d}: {file} \t| Data shape: {data.shape} \t| {attrs_str}")
        from collections import Counter
        print('Total points:', point_number)
        print('Total labels:', len(set(self.file_label)), ' :', Counter(self.file_label))
        print('Total conditions:', len(set(self.file_condition)), ' :', set(self.file_condition))
        return None
    
    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]

        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.file_label[file_idx]
        condition = self.file_condition[file_idx]
        channel_names = self.file_channel_names[file_idx]

        return {'data': data, 'label': label, 'condition': condition, 'channel_names': channel_names}
    
    def __len__(self):
        return self.length
    
    @property
    def file_num(self):
        return len(self.file_list)
    
class Dataset_CWRU(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='CWRU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_CWRU, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'B007': 0,
                'B014': 1,
                'B021': 2,
                'B028': 3,
                'IR007': 4,
                'IR014': 5,
                'IR021': 6,
                'IR028': 7,
                'OR007_3': 8,
                'OR007_6': 9,
                'OR007_12': 10,
                'OR014_6': 11,
                'OR021_3': 12,
                'OR021_6': 13,
                'OR021_12': 14,
                'N': 15,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'N': 0,
                'IR007': 1,
                'IR014': 1,
                'IR021': 1,
                'IR028': 1,
                'OR007_3': 2,
                'OR007_6': 2,
                'OR007_12': 2,
                'OR014_6': 2,
                'OR021_3': 2,
                'OR021_6': 2,
                'OR021_12': 2,
                'B007': 3,
                'B014': 3,
                'B021': 3,
                'B028': 3,
            }
            self.num_classes = 4
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition

class Dataset_HUST_bearing(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='HUST_bearing.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_HUST_bearing, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'B_2': 0,
                'C_2': 1,
                'I_2': 2,
                'O_2': 3,
                'B': 4,
                'C': 5,
                'I': 6,
                'O': 7,
                'H': 8,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'H': 0,
                'I': 1,
                'I_2': 1,
                'O': 2,
                'O_2': 2,
                'B': 3,
                'B_2': 3,
                'C': 4,
                'C_2': 4  
            }
            self.num_classes = 5
        self.__pad_channel__()
    

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition

class Dataset_XJTU_CLS(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='XJTU_CLS.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_XJTU_CLS, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            
            self.label_code = {

                'IR': 0,
                'OR': 1,
                'IR_OR': 2,
                'cage': 3,
                'IR_B_cage_OR': 4
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'IR': 1,
                'OR': 2,
                'IR_OR': 3,
                'cage': 4,
                'IR_B_cage_OR': 5
            }
            self.num_classes = 5
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_HUST_gearbox(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='HUST_gearbox.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_HUST_gearbox, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'C': 0,
                'H': 1,
                'M': 2,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'C': 0,
                'H': 1,
                'M': 2,
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_IMS(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='IMS.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_IMS, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'OR': 2,
                'IR': 1,
                'Normal': 0,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'OR': 2,
                'IR': 1,
                'Normal': 0,
            }
            self.num_classes = 3
        self.__pad_channel__()
    

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition

class Dataset_LW(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='LW.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_LW, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            self.label_code = {
                'Normal': 0,
                '5mm crack': 1,
                '15mm crack': 2,
                '10mm crack': 3,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'Normal': 0,
                '5mm crack': 1,
                '15mm crack': 2,
                '10mm crack': 3,
            }
            self.num_classes = 4
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_PU(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='PU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_PU, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'IR_1_EDM': 0,
                'OR_1_EMD': 1,
                'OR_1_drilling':2,
                'IR_1_electric_engraver':3,
                'OR_1_electric_engraver':4,
                'IR_1_pitting': 5,
                'OR_1_pitting': 6,
                'OR_2_drilling':7,
                'IR_2_electric_engraver':8,
                'OR_2_electric_engraver':9,
                'IR_2_pitting': 10,
                'OR_2_pitting': 11,

                
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'IR_1_EDM': 0,
                'OR_1_EMD': 1,
                'OR_1_drilling':2,
                'IR_1_electric_engraver':3,
                'OR_1_electric_engraver':4,
                'IR_1_pitting': 5,
                'OR_1_pitting': 6,
                'OR_2_drilling':2,
                'IR_2_electric_engraver':3,
                'OR_2_electric_engraver':4,
                'IR_2_pitting': 5,
                'OR_2_pitting': 6,
            }
            self.num_classes = 7
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_SCP(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='SCP.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_SCP, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'IR': 1,
                'OR': 2,
                'Normal':0,
   
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'IR': 1,
                'OR': 2,
                'Normal':0
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_JNU(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='JNU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_JNU, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'IR': 1,
                'OR': 2,
                'B':3,
                'N':0
   
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'IR': 1,
                'OR': 2,
                'B':3,
                'N':0
            }
            self.num_classes = 4
        self.__pad_channel__()


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition

    
class Dataset_PHM2009(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='PHM2009.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_PHM2009, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            
            self.label_code = {
                'N': 0,
                'In_Ks': 1,
                'E':2,
                'C_E_Br_In_B_O':3,
                'C_E':4,
                'Br_In_B_O_Im':5,
                'Br_B':6,
                'B_O_Im':7,

   
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'N': 0,
                'In_Ks': 1,
                'E':2,
                'C_E_Br_In_B_O':3,
                'C_E':4,
                'Br_In_B_O_Im':5,
                'Br_B':6,
                'B_O_Im':7,
            }
            self.num_classes = 8
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_QPZZ(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='QPZZ.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_QPZZ, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,few_shot=few_shot)
        
        if label_type == 'local' or label_type == 'global':
            self.label_code = {
                'normal': 0,
                'broken': 1,
                'wear': 2,
                'pitting': 3,
                'pitting_wear': 4,
            }
            self.num_classes = len(self.label_code)
        # if label_type == 'global':
        #     self.label_code = self.global_label_index
        #     self.num_classes = 5

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition

    
    
class Dataset_SLIET(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='SLIET.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_SLIET, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,few_shot=few_shot)

        if label_type == 'local':
            self.label_code = {
                'N': 0,
                'IR_1': 1,
                'IR_2': 2,
                'IR_3': 3,
                'IR_4': 4,
                'OR_1': 5,
                'OR_2': 6,
                'OR_3': 7,
                'OR_4': 8,
                'RO_1': 9,
                'RO_2': 10,
                'RO_3': 11,
                'RO_4': 12,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            self.label_code = {
                'N': 0,
                'IR_1': 1,
                'IR_2': 1,
                'IR_3': 1,
                'IR_4': 1,
                'OR_1': 2,
                'OR_2': 2,
                'OR_3': 2,
                'OR_4': 2,
                'RO_1': 3,
                'RO_2': 3,
                'RO_3': 3,
                'RO_4': 3,
            }            
            self.num_classes = 4

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition

    
class Dataset_SMU(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='SMU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_SMU, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,few_shot=few_shot)

        if label_type == 'local' or label_type == 'global':
            self.label_code = {
                'normal': 0,
                'wear': 1,
                'chip': 2,
            }
            self.num_classes = len(self.label_code)
        # if label_type == 'global':
        #     self.label_code = self.global_label_index
        #     self.num_classes = 3

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_ROT(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='ROT.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None,few_shot = False):
        super(Dataset_ROT, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                          down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3,few_shot=few_shot)
        if label_type == 'local':
            # Labels present in ROT.hdf5: Normal, Runout, Gear_NG
            self.label_code = {
                'Normal': 0,
                'Runout': 1,
                'Bearing_NG': 2,
                'Gear_NG': 3,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # Same mapping for simplicity; adjust if needed
            self.label_code = {
                'Normal': 0,
                'Runout': 1,
                'Bearing_NG': 2,
                'Gear_NG': 3,
            }
            self.num_classes = 4
        self.__pad_channel__()

    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])
        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        return data, label,condition


class Dataset_XJTU(Dataset):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, num_look_back=8, flag=None, test_idx=None,
                 down_sampling_scale=1, data_path='XJTU_2.hdf5', start_percentage=0.0, end_percentage=1.0,label_type=None):

        '''
        ********** XJTU_RUL Bearing List **********
        # 打#的轴承剔除掉，不参与训练和测试
        # [0] 2100rpm_12kN_2h2min_bearing1_4      112
        [0] 2100rpm_12kN_2h38min_bearing1_3     56
        [1] 2100rpm_12kN_2h3min_bearing1_1      75
        [2] 2100rpm_12kN_2h41min_bearing1_2     45
        [3] 2100rpm_12kN_52min_bearing1_5       31
        [4] 2250rpm_11kN_2h41min_bearing2_2     47
        [5] 2250rpm_11kN_42min_bearing2_4       24
        [6] 2250rpm_11kN_5h39min_bearing2_5     145
        [7] 2250rpm_11kN_8h11min_bearing2_1     450
        [8] 2250rpm_11kN_8h53min_bearing2_3     290
        [9] 2400rpm_10kN_1h54min_bearing3_5    8
        [10] 2400rpm_10kN_25h15min_bearing3_4   1428
        # [12] 2400rpm_10kN_41h36min_bearing3_2   2069
        [11] 2400rpm_10kN_42h18min_bearing3_1   2358
        [12] 2400rpm_10kN_6h11min_bearing3_3    338
        '''
        #如果seq_len不被num_look_back整除，报错
        assert seq_len % num_look_back == 0, 'seq_len should be divisible by num_look_back'
        self.__root_path = root_path
        self.__data_path = data_path
        self.__file_path = Path(self.__root_path) / self.__data_path
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.num_look_back = num_look_back
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.flag = flag
        self.test_idx = test_idx

        self.feature_size = None
        self.channel_names = None
        self.length = None

        self.condition_code = {}
        self.channel_code = {}
        self.fault_code = {}

        # 以下定义的列表用于存放后处理的数据切片，索引，标签，工况，故障元素等信息
        # 所有列表的长度均一致，且一一对应
        self.file_list = []
        self.global_idxes = []
        self.local_idxes = []
        self.file_name = []
        self.file_data = []
        self.file_label = []
        self.file_condition = []
        self.file_fault_element = []
        self.num_classes = 4
        # 以下定义的字典用于存放文件名和其对应的寿命、自适应步长、初始预测时间FPT
        # 均是以字典的形式访问，key为文件名，仅在__read_data__中使用
        self.file_life = {}
        self.adaptive_stride = {}
        self.subject_fpt = {
            '2100rpm_12kN_2h2min_bearing1_4': 112,
            '2100rpm_12kN_2h38min_bearing1_3': 56,
            '2100rpm_12kN_2h3min_bearing1_1': 75,
            '2100rpm_12kN_2h41min_bearing1_2': 45,
            '2100rpm_12kN_52min_bearing1_5': 31,
            '2250rpm_11kN_2h41min_bearing2_2': 47,
            '2250rpm_11kN_42min_bearing2_4': 24,
            '2250rpm_11kN_5h39min_bearing2_5': 145,
            '2250rpm_11kN_8h11min_bearing2_1': 450,
            '2250rpm_11kN_8h53min_bearing2_3': 290,
            '2400rpm_10kN_1h54min_bearing3_5': 8,
            '2400rpm_10kN_25h15min_bearing3_4': 1428,
            '2400rpm_10kN_41h36min_bearing3_2': 2069,
            '2400rpm_10kN_42h18min_bearing3_1': 2358,
            '2400rpm_10kN_6h11min_bearing3_3': 338,
        }

        self.__read_data__()
        self.__standardize__()

    def __standardize__(self):
        # Standardize the data
        # 讲所有的数据进行标准化
        self.scaler = StandardScaler()
        x_data_all = []
        for i in range(len(self.file_data)):
            x_data_all.append(self.file_data[i])
        x_data_all = np.vstack(x_data_all)
        self.scaler.fit(x_data_all)
        for i in range(len(self.file_data)):
            self.file_data[i] = torch.tensor(self.scaler.transform(self.file_data[i]), dtype=torch.float32)
        return None


    def __read_data__(self):
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.file_list = [i for i in self.__file]
        self.file_list.remove('2100rpm_12kN_2h2min_bearing1_4')
        self.file_list.remove('2400rpm_10kN_41h36min_bearing3_2')
        self.file_life = {file: self.format_life_time(self.__file[file].attrs['life_time']) for file in self.file_list}
        self.degrade_len = {file: self.file_life[file] - self.subject_fpt[file] + 1 for file in self.file_list}

        self.feature_size = [self.seq_len, self.__file[self.file_list[0]][:].shape[1]]
        self.one_sample_length = int(25600 * 1.28)
        self.down_sampled_one_sample_length = self.one_sample_length // self.down_sampling_scale
        self.adaptive_stride = self.get_adaptive_stride()
        self.cut_length_at_one_life_time = self.seq_len // self.num_look_back

        if self.flag == 'pretrain':
            current_file_list = self.file_list
        elif self.flag == 'train':
            current_file_list = [subject for i, subject in enumerate(self.file_list) if i not in self.test_idx]
        elif self.flag == 'test':
            current_file_list = [subject for i, subject in enumerate(self.file_list) if i in self.test_idx]

        global_index = 0
        total_point_num = 0
        for file in current_file_list:

            life_label = np.concatenate([np.ones(self.subject_fpt[file] - 1), np.linspace(1, 0, self.file_life[file] - self.subject_fpt[file] + 1)])
            life_label = torch.tensor(life_label.repeat(self.one_sample_length), dtype=torch.float32)
            data = self.__file[file][:]
            total_point_num += data.shape[0]
            if self.flag in ['train', 'test']:
                data = data[(self.subject_fpt[file] - self.num_look_back) * self.one_sample_length:]
                life_label = life_label[(self.subject_fpt[file] - self.num_look_back) * self.one_sample_length:]
            elif self.flag == 'pretrain':
                pass

            # 去除每个sample文件降采样后剩余的部分, 保证每部分降采样后的序列取出的样本数相同
            data = data.reshape(-1, self.one_sample_length, data.shape[-1])
            life_label = life_label.reshape(-1, self.one_sample_length)
            data = data[:, :self.down_sampling_scale * self.down_sampled_one_sample_length, :]
            life_label = life_label[:, :self.down_sampling_scale * self.down_sampled_one_sample_length]
            data = data.reshape(-1, data.shape[-1])
            life_label = life_label.reshape(-1)

            for i in range(self.down_sampling_scale):
                # 降采样后的数据
                self.global_idxes.append(global_index)
                data_i = data[i::self.down_sampling_scale]
                label_i = life_label[i::self.down_sampling_scale]
                file_length = data_i.shape[0] - (self.num_look_back - 1) * self.down_sampled_one_sample_length
                self.file_name.append(file)
                self.file_data.append(data_i)
                self.file_label.append(label_i)
                self.file_condition.append(self.__file[file].attrs['condition'])

                total_sample_num = (file_length - self.seq_len) // self.adaptive_stride[file] + 1
                start_idx = int(total_sample_num * self.start_percentage) * self.adaptive_stride[file]
                end_idx = int(total_sample_num * self.end_percentage - 1) * self.adaptive_stride[file]

                self.local_idxes.append(start_idx)
                global_index += (end_idx - start_idx) // self.adaptive_stride[file] + 1
        self.length = global_index
        self.condition_code = {condition: idx for idx, condition in enumerate(sorted(set(self.file_condition)))}

        # 生成一组初始的mask索引，即如果self.num_look_back=8, 则生成8组256长度的的索引，
        # 每组之间间隔为self.down_sampled_one_sample_length = 6553, 总长度为self.seq_len=2048
        self.init_mask_indexes = torch.tensor(np.concatenate([np.arange(256) + i * self.down_sampled_one_sample_length for i in range(self.num_look_back)]), dtype=torch.long)

        # self.__file.close()
        print('Total points:', total_point_num)
        return None
    
    
    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.adaptive_stride[self.file_name[file_idx]] + self.local_idxes[file_idx]

        # 检查item_start_idx是否针对长度为self.down_sampled_one_sample_length的同寿命样本存在溢出
        overflow = (item_start_idx + self.cut_length_at_one_life_time) % self.down_sampled_one_sample_length

        # 如果溢出，向后移直到不溢出，保证label的一致性
        item_start_idx = item_start_idx if overflow >= self.cut_length_at_one_life_time else item_start_idx + self.cut_length_at_one_life_time - overflow

        # 根据item_start_idx重新定位mask索引
        mask_indexes = self.init_mask_indexes + item_start_idx

        # 根据mask从原序列中取数据和标签，标签取最后一个时刻的寿命
        data = torch.index_select(self.file_data[file_idx], 0, mask_indexes)
        label = torch.index_select(self.file_label[file_idx], 0, mask_indexes[-self.cut_length_at_one_life_time:])
        condition = torch.tensor([self.condition_code[self.file_condition[file_idx]]])

        if self.flag in ['pretrain', 'train']:
            # return {'data': data, 'label': label, 'condition': condition, 'fault_element': fault_element}
            return data, label[0],condition
        elif self.flag == 'test':
            name_and_id = (self.file_name[file_idx], idx - self.global_idxes[file_idx])
            # return {'data': data, 'label': label, 'condition': condition, 'fault_element': fault_element, 'name_and_id': name_and_id}
            return data, label[0], condition,name_and_id
        
        
    def __len__(self):
        return self.length
    


    def get_adaptive_stride(self):

        min_degrade_len = min(self.degrade_len.values())
        total_num_sample_of_one_down_sampled_sample = (self.down_sampled_one_sample_length * min_degrade_len - self.seq_len) // self.stride_len + 1
        adaptive_stride = {}

        for file, degrade_len in self.degrade_len.items():
            if degrade_len == min_degrade_len:
                adaptive_stride[file] = self.stride_len
            else:
                adaptive_stride[file] = (degrade_len * self.down_sampled_one_sample_length - self.seq_len) // (total_num_sample_of_one_down_sampled_sample + 1)
        return adaptive_stride


    def format_life_time(self, life_time_str):

        if 'h' not in life_time_str:
            life_time_str = '0h' + life_time_str
        hour = int(life_time_str.split('h')[0])
        minute = int(life_time_str.split('h')[1][:-3])
        life = hour * 60 + minute
        return life



class Dataset_XJTU_old(Dataset):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, flag=None, test_idx=None,
                 down_sampling_scale=1, data_path='XJTU_2.hdf5', start_percentage=0.0, end_percentage=1.0,label_type=None):

        '''
        ********** XJTU_RUL Bearing List **********
        # 打#的轴承剔除掉，不参与训练和测试
        # [0] 2100rpm_12kN_2h2min_bearing1_4      112
        [0] 2100rpm_12kN_2h38min_bearing1_3     56
        [1] 2100rpm_12kN_2h3min_bearing1_1      75
        [2] 2100rpm_12kN_2h41min_bearing1_2     45
        [3] 2100rpm_12kN_52min_bearing1_5       31
        [4] 2250rpm_11kN_2h41min_bearing2_2     47
        [5] 2250rpm_11kN_42min_bearing2_4       24
        [6] 2250rpm_11kN_5h39min_bearing2_5     145
        [7] 2250rpm_11kN_8h11min_bearing2_1     450
        [8] 2250rpm_11kN_8h53min_bearing2_3     290
        [9] 2400rpm_10kN_1h54min_bearing3_5    8
        [10] 2400rpm_10kN_25h15min_bearing3_4   1428
        # [12] 2400rpm_10kN_41h36min_bearing3_2   2069
        [11] 2400rpm_10kN_42h18min_bearing3_1   2358
        [12] 2400rpm_10kN_6h11min_bearing3_3    338
        '''
        
        self.__root_path = root_path
        self.__data_path = data_path
        self.__file_path = Path(self.__root_path) / self.__data_path
        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.flag = flag
        self.test_idx = test_idx

        self.feature_size = None
        self.channel_names = None
        self.length = None

        self.condition_code = {}
        self.channel_code = {}
        self.fault_code = {}

        # 以下定义的列表用于存放后处理的数据切片，索引，标签，工况，故障元素等信息
        # 所有列表的长度均一致，且一一对应
        self.file_list = []
        self.global_idxes = []
        self.local_idxes = []
        self.file_name = []
        self.file_data = []
        self.file_label = []
        self.file_condition = []
        self.file_fault_element = []

        # 以下定义的字典用于存放文件名和其对应的寿命、自适应步长、初始预测时间FPT
        # 均是以字典的形式访问，key为文件名，仅在__read_data__中使用
        self.file_life = {}
        self.adaptive_stride = {}
        self.subject_fpt = {
            '2100rpm_12kN_2h2min_bearing1_4': 112,
            '2100rpm_12kN_2h38min_bearing1_3': 56,
            '2100rpm_12kN_2h3min_bearing1_1': 75,
            '2100rpm_12kN_2h41min_bearing1_2': 45,
            '2100rpm_12kN_52min_bearing1_5': 31,
            '2250rpm_11kN_2h41min_bearing2_2': 47,
            '2250rpm_11kN_42min_bearing2_4': 24,
            '2250rpm_11kN_5h39min_bearing2_5': 145,
            '2250rpm_11kN_8h11min_bearing2_1': 450,
            '2250rpm_11kN_8h53min_bearing2_3': 290,
            '2400rpm_10kN_1h54min_bearing3_5': 8,
            '2400rpm_10kN_25h15min_bearing3_4': 1428,
            '2400rpm_10kN_41h36min_bearing3_2': 2069,
            '2400rpm_10kN_42h18min_bearing3_1': 2358,
            '2400rpm_10kN_6h11min_bearing3_3': 338,
        }

        self.__read_data__()
        self.__standardize__()

    def __standardize__(self):
        # Standardize the data
        # 讲所有的数据进行标准化
        self.scaler = StandardScaler()
        x_data_all = []
        for i in range(len(self.file_data)):
            x_data_all.append(self.file_data[i])
        x_data_all = np.vstack(x_data_all)
        self.scaler.fit(x_data_all)
        for i in range(len(self.file_data)):
            self.file_data[i] = torch.tensor(self.scaler.transform(self.file_data[i]), dtype=torch.float32)
        return None


    def __read_data__(self):
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.file_list = [i for i in self.__file]
        self.file_list.remove('2100rpm_12kN_2h2min_bearing1_4')
        self.file_list.remove('2400rpm_10kN_41h36min_bearing3_2')
        self.file_life = {file: self.format_life_time(self.__file[file].attrs['life_time']) for file in self.file_list}
        self.degrade_len = {file: self.file_life[file] - self.subject_fpt[file] + 1 for file in self.file_list}

        self.feature_size = [self.seq_len, self.__file[self.file_list[0]][:].shape[1]]
        self.one_sample_length = int(25600 * 1.28)
        self.down_sampled_one_sample_length = self.one_sample_length // self.down_sampling_scale
        self.adaptive_stride = self.get_adaptive_stride()

        if self.flag == 'pretrain':
            current_file_list = self.file_list
        elif self.flag == 'train':
            current_file_list = [subject for i, subject in enumerate(self.file_list) if i not in self.test_idx]
        elif self.flag == 'test':
            current_file_list = [subject for i, subject in enumerate(self.file_list) if i in self.test_idx]

        global_index = 0
        for file in current_file_list:

            life_label = np.concatenate([np.ones(self.subject_fpt[file] - 1), np.linspace(1, 0, self.file_life[file] - self.subject_fpt[file] + 1)])
            life_label = life_label.repeat(self.one_sample_length)
            data = self.__file[file][:]

            if self.flag in ['train', 'test']:
                data = data[(self.subject_fpt[file] - 1) * self.one_sample_length:]
                life_label = life_label[(self.subject_fpt[file] - 1) * self.one_sample_length:]
            elif self.flag == 'pretrain':
                pass

            # 去除每个sample文件降采样后剩余的部分, 保证每部分降采样后的序列取出的样本数相同
            data = data.reshape(-1, self.one_sample_length, data.shape[-1])
            life_label = life_label.reshape(-1, self.one_sample_length)
            data = data[:, :self.down_sampling_scale * self.down_sampled_one_sample_length, :]
            life_label = life_label[:, :self.down_sampling_scale * self.down_sampled_one_sample_length]
            data = data.reshape(-1, data.shape[-1])
            life_label = life_label.reshape(-1)

            for i in range(self.down_sampling_scale):
                # 降采样后的数据
                self.global_idxes.append(global_index)
                data_i = data[i::self.down_sampling_scale]
                label_i = life_label[i::self.down_sampling_scale]
                file_length = data_i.shape[0]
                self.file_name.append(file)
                self.file_data.append(data_i)
                self.file_label.append(label_i)
                self.file_condition.append(self.__file[file].attrs['condition'])

                total_sample_num = (file_length - self.seq_len) // self.adaptive_stride[file] + 1
                start_idx = int(total_sample_num * self.start_percentage) * self.adaptive_stride[file]
                end_idx = int(total_sample_num * self.end_percentage - 1) * self.adaptive_stride[file]

                self.local_idxes.append(start_idx)
                global_index += (end_idx - start_idx) // self.adaptive_stride[file] + 1
        self.length = global_index

        # self.__file.close()

        return None
    
    
    def __getitem__(self, idx: int):
        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.adaptive_stride[self.file_name[file_idx]] + self.local_idxes[file_idx]

        label = self.file_label[file_idx][item_start_idx:item_start_idx + self.seq_len]
        if len(set(label)) > 1:
            change_point = np.where(np.diff(label) != 0)[0][0]
            if change_point > self.seq_len // 2:
                item_start_idx = item_start_idx - (self.seq_len - change_point - 1)
            else:
                item_start_idx = item_start_idx + change_point + 1

            data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
            label = self.file_label[file_idx][item_start_idx:item_start_idx + self.seq_len]
        else:
            data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        # condition = self.condition_code[self.file_condition[file_idx]]
        # fault_element = self.fault_code[self.file_fault_element[file_idx]]

        if self.flag in ['pretrain', 'train']:
            # return {'data': data, 'label': label, 'condition': condition, 'fault_element': fault_element}
            return data, label[0]
        elif self.flag == 'test':
            name_and_id = (self.file_name[file_idx], idx - self.global_idxes[file_idx])
            # return {'data': data, 'label': label, 'condition': condition, 'fault_element': fault_element, 'name_and_id': name_and_id}
            return data, label[0], name_and_id
        
        
    def __len__(self):
        return self.length
    


    def get_adaptive_stride(self):

        min_degrade_len = min(self.degrade_len.values())
        total_num_sample_of_one_down_sampled_sample = (self.down_sampled_one_sample_length * min_degrade_len - self.seq_len) // self.stride_len + 1
        adaptive_stride = {}

        for file, degrade_len in self.degrade_len.items():
            if degrade_len == min_degrade_len:
                adaptive_stride[file] = self.stride_len
            else:
                adaptive_stride[file] = (degrade_len * self.down_sampled_one_sample_length - self.seq_len) // (total_num_sample_of_one_down_sampled_sample + 1)
        return adaptive_stride


    def format_life_time(self, life_time_str):

        if 'h' not in life_time_str:
            life_time_str = '0h' + life_time_str
        hour = int(life_time_str.split('h')[0])
        minute = int(life_time_str.split('h')[1][:-3])
        life = hour * 60 + minute
        return life




def pad_and_stack(arrays):
    """
    Pads and stacks a list of numpy arrays of varying lengths and creates a boolean array 
    indicating padded elements.

    Args:
    arrays (list of np.array): List of one-dimensional numpy arrays.

    Returns:
    np.array: A two-dimensional numpy array where each original array is padded with zeros
              to match the length of the longest array in the list.
    np.array: A two-dimensional boolean array where True indicates a padded element and 
              False indicates an original element.
    """
    # Find the maximum length among all arrays
    max_len = max(len(a) for a in arrays)

    # Initialize lists to hold padded arrays and boolean arrays
    padded_arrays = []
    boolean_arrays = []

    for a in arrays:
        # Amount of padding needed
        padding = max_len - len(a)

        # Pad the array and add it to the list
        padded_arrays.append(np.pad(a, (0, padding), mode='constant'))

        # Create a boolean array (False for original elements, True for padding)
        boolean_array = np.array([False] * len(a) + [True] * padding)
        boolean_arrays.append(boolean_array)

    # Stack the padded arrays and boolean arrays vertically
    stacked_array = np.vstack(padded_arrays)
    boolean_stacked = np.vstack(boolean_arrays)

    return stacked_array, boolean_stacked


def context_based_split(X, is_pad, context_len: int):
    split_inds = np.arange(start=0, stop=X.shape[1], step=context_len)

    X_collapse = []
    is_pad_collapse = []

    for i in range(1, len(split_inds)):
        X_collapse.append(X[:, split_inds[i-1]:split_inds[i]])
        is_pad_collapse.append(is_pad[:, split_inds[i-1]:split_inds[i]])

    Xnew = np.concatenate(X_collapse, axis=0)
    pad_new = np.concatenate(is_pad_collapse, axis=0)

    pad_by_sample = np.any(pad_new, axis=1)

    return Xnew[~pad_by_sample, :]

