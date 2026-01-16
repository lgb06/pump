from data_provider.data_loader import *
from data_provider.DG_data_loader import *
from ts_lib_data_provider.data_loader import UEAloader
from ts_lib_data_provider.uea import collate_fn as uea_collate_fn
import torch
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler

from utils.ddp import is_main_process

data_dict = {
    'CWRU': Dataset_CWRU,
    'QPZZ': Dataset_QPZZ,
    'SLIET': Dataset_SLIET,
    'SMU': Dataset_SMU,
    'ROT': Dataset_ROT,
    'HUST_gearbox': Dataset_HUST_gearbox,
    'HUST_bearing': Dataset_HUST_bearing,
    'IMS': Dataset_IMS,
    'SCP': Dataset_SCP,
    'PU': Dataset_PU,
    'LW': Dataset_LW,
    'JNU': Dataset_JNU,
    'RUL_XJTU': Dataset_XJTU,
    ##DG
    'DG_CWRU': Dataset_CWRU_DG,
    'DG_IMS': Dataset_IMS_DG,
    'DG_SCP': Dataset_SCP_DG,
    'DG_JNU': Dataset_JNU_DG,
    'DG_HUST_bearing': Dataset_HUST_bearing_DG,
    'NLNEMP': NLNEMPloader,
    'NLNEMP_Elec': NLNEMP_ElecLoader,
    'NLNEMP_Vib': NLNEMP_VibLoader,
    'UEA': UEAloader,
    # /*TODO*/
}   

import random
from collections import defaultdict


def data_provider(args, config, flag, ddp=False):  # args,
    Data = data_dict[config['data']]
    timeenc = 0 if config['embed'] != 'timeF' else 1
    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if 'anomaly_detection' in config['task_name']:  # working on one gpu
            batch_size = args.batch_size
        else:
        #增大batch_size，加速测试时间
            batch_size = max(256,args.batch_size)
        freq = args.freq
    elif flag == 'pretrain':
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    else:
        shuffle_flag = False if 'RUL' in config['task_name'] else True
        # shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if 'PHM' in config['task_name']:
        if 'RUL' in config['task_name']:
            data_set = Data(
                root_path=config['root_path'],
                seq_len=config['seq_len'],
                stride_len=config['stride_len'],
                down_sampling_scale=config['down_sample'],
                flag = flag,
                test_idx = config['test_idx'],
                args = args
            )
        elif  args.task_name =='few_shot' and flag == 'train':
            data_set = Data(
                root_path=config['root_path'],
                seq_len=config['seq_len'],
                stride_len=config['stride_len'],
                down_sampling_scale=config['down_sample'],
                label_type=config['label_type'] if 'label_type' in config else None,
                flag = flag,
                args = args,
                few_shot = config['few_shot']
            )
        else:
            data_set = Data(
                root_path=config['root_path'],
                seq_len=config['seq_len'],
                stride_len=config['stride_len'],
                down_sampling_scale=config['down_sample'],
                label_type=config['label_type'] if 'label_type' in config else None,
                flag = flag,
                args = args
            )
        if args.task_name =='few_shot' and flag == 'train':

            print("few_shot at setiing of", len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=drop_last)

        return data_set, data_loader
    elif 'classification' in config['task_name'] and config.get('data') == 'UEA':
        dataset_name = config.get('dataset_name', config.get('data'))
        data_set = Data(
            args=args,
            root_path=config['root_path'],
            flag=flag,
            file_list=config.get('file_list'),
            limit_size=config.get('limit_size'),
            dataset_name=dataset_name,
        )

        max_len = config.get('seq_len', getattr(args, 'input_len', None))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=False if ddp else shuffle_flag,
            num_workers=args.num_workers,
            sampler=DistributedSampler(data_set) if ddp else None,
            drop_last=False,
            collate_fn=lambda x: uea_collate_fn(x, max_len=max_len, num_classes=data_set.num_classes)
        )

        return data_set, data_loader
