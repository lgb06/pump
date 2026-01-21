import torch
import bisect
import sys
import os
import glob
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
sys.path.append('/dataWYL/WYL/PHM-Large-Model')
# from data_provider.m4 import M4Dataset, M4Meta # removed due to 
import warnings
import torch.nn.functional as F
from data_provider.data_loader import Dataset_PHM
# warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from collections import Counter



class Dataset_CWRU_DG(Dataset_PHM):
    def __init__(self, root_path, args,seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='DG_CWRU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_CWRU_DG, self).__init__(root_path, args ,seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
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
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        # label = F.one_hot(torch.tensor(label), num_classes=self.num_classes)
        label = F.one_hot(torch.tensor(label), num_classes=2)

        print(f"data.shape{data.shape}")

        return data, label,label

class Dataset_HUST_bearing_DG(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='DG_HUST_bearing.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_HUST_bearing_DG, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            
            self.label_code = {
               'H': 0,
                'I': 1,
                'I_2': 2,
                'O': 3,
                'O_2': 4,
                'B': 5,
                'B_2': 6
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
                'B_2': 3
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=2)

        return data, label,label
class Dataset_XJTU_CLS_DG(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='DG_XJTU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_XJTU_CLS_DG, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            
            self.label_code = {
                'IR': 1,
                'OR': 2
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'IR': 1,
                'OR': 2
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=2)

        return data, label,label

class Dataset_IMS_DG(Dataset_PHM):
    def __init__(self, root_path, args,seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='DG_IMS.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_IMS_DG, self).__init__(root_path, args, seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            
            self.label_code = {
                'Normal': 0,
                'IR': 1,
                'OR': 2,
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'Normal': 0,
                'IR': 1,
                'OR': 2,
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=2)

        return data, label,label

class Dataset_SCP_DG(Dataset_PHM):
    def __init__(self, root_path, args ,seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='DG_SCP.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_SCP_DG, self).__init__(root_path, args ,seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            
            self.label_code = {
                'Normal':0,
                'IR': 1,
                'OR': 2
   
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'Normal':0,
                'IR': 1,
                'OR': 2
            }
            self.num_classes = 3
        self.__pad_channel__()
    


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=2)

        return data, label,label

class Dataset_JNU_DG(Dataset_PHM):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, label_type='local', cross_condition=None,
                 down_sampling_scale=1, data_path='DG_JNU.hdf5', start_percentage=0.0, end_percentage=1.0, flag=None):
        super(Dataset_JNU_DG, self).__init__(root_path, args,  seq_len, stride_len, flag, cross_condition,
                                           down_sampling_scale, data_path, start_percentage, end_percentage,padd_channel_num=3)
        if label_type == 'local':
            
            self.label_code = {
                'N':0,
                'IR': 1,
                'OR': 2
            }
            self.num_classes = len(self.label_code)
        if label_type == 'global':
            # self.label_code = self.global_label_index
            self.label_code = {
                'N':0,
                'IR': 1,
                'OR': 2
            }
            self.num_classes = 3
        self.__pad_channel__()


    def __getitem__(self, idx: int):

        file_idx = bisect.bisect(self.global_idxes, idx) - 1
        item_start_idx = (idx - self.global_idxes[file_idx]) * self.stride_len + self.local_idxes[file_idx]
        data = self.file_data[file_idx][item_start_idx:item_start_idx + self.seq_len]
        label = self.label_code[self.file_label[file_idx]]

        # 将离散的标签转换为one-hot向量
        # 二分类
        if label != 0:  label = 1
        label = F.one_hot(torch.tensor(label), num_classes=2)

        return data, label,label
    
# padding_mask 只是用来告诉模型哪些位置是真实数据、哪些是 padding   
# data_factory.py 对 PHM 任务直接把 Dataset 的三个返回值组成 batch，不再做自定义 collate。（exp_pretrain.py/exp_sup.py 里的 PHM 处理分支（如 get_multi_source_data）期望三元组 (batch_x, batch_x_mark, padding_mask)。）
# PHM 数据本身所有位置都是真实数据，不涉及padding，所以实际上不需要padding mask。所以这里用 label 作为占位传上去，保持三元组长度不报错。
# 模型实际不会把这个占位的第三个值当作真实标签用；在分类路径里，它通常只是被转换成全 1 mask 或直接忽略。




# /*TODO*/
class NLNEMPloader(Dataset):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, down_sampling_scale=1, label_type='local', flag=None, modality=None, motor_id='all', normalize=True, transform=None):

                 # /*TODO*/ flag 区分测试与训练
        """
        参数:
        - root_path: 数据集根目录
        - modality: 'Vibration' 或 'Electric'
        - motor_id: 电机 ID
        - down_sampling_scale: 降采样率 (默认为10，即20kHz -> 2kHz)
        - normalize: 是否对数据进行 Z-Score 标准化 (均值0，方差1)
        - transform: 可选的额外的预处理函数 (如数据增强)
        """
        super().__init__()
        self.args = args
        self.root_path = root_path
        if modality is None:
            self.modality = "Electric" if self.args.data == "electric" or self.args.data == "Electric" else 'Vibration'
        else:
            self.modality = modality

        self.num_classes = args.num_classes 
        self.project_name = args.project_name      
        self.data_demo = getattr(args, "data_demo", False)

        self.seq_len = seq_len
        self.stride_len = stride_len
        self.down_sampling_scale = down_sampling_scale
        self.label_type = label_type
        self.flag = flag

        
        self.transform = transform
        self.normalize = normalize
        
        # 1. 加载数据到内存
        print(f"加载 NLN-EMP Dataset: {motor_id} - {self.modality} (降采样: {down_sampling_scale})")
        self.data, self.labels, self.label_map = self._load_all_data(
            flag, self.args.cls_mode ,self.modality, motor_id, down_sampling_scale, root_path 
        )

        # /*TODO*/ flag 数据截断！
        # if flag=='TRAIN':            # 把y给截断没了，self.labels = []
        #     self.data = self.data[:len(self.data)//5]
        #     self.labels = self.labels[:len(self.data)//5]
        # elif flag=='TEST': 
        #     self.data = self.data[len(self.data)//5:]
        #     self.labels = self.labels[len(self.data)//5:]

        # print(f"self.labels:{self.labels}")
        # print(f"len(self.data):{len(self.data)}")

        # self.data = self.data[len(self.data)//5:]
        # self.labels = self.labels[len(self.data)//5:]

        # print(f"self.labels:{self.labels}")

        

        # use all features
        self.feature_df = self.data
        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)


        
        # # 2. 计算统计量并标准化 (如果在训练集上计算，应保存mean/std应用到测试集)
        # if self.normalize and len(self.data) > 0:
        #     # 计算全局均值和标准差 (按通道计算)
        #     # data shape: (N, T, C) -> 这里的 axis=(0,1) 表示在 样本和时间维度上聚合，保留通道维度
        #     self.mean = np.mean(self.data, axis=(0, 1))
        #     self.std = np.std(self.data, axis=(0, 1)) + 1e-8 # 加小数值防止除以0
        #     print(f"数据标准化开启: Mean={self.mean}, Std={self.std}")
        # else:
        #     self.mean = None
        #     self.std = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        返回:
        - x: Tensor, 形状 (Time_Steps, Channels) 
        - y: Tensor, 标量标签 (Long)
        """


        # 获取原始数据 (Time, Channel)
        # x_np = self.data[idx]
        x_np = self.feature_df[idx]
        y_val = self.labels[idx]

        # # 标准化
        # if self.normalize:
        #     x_np = (x_np - self.mean) / self.std

        # 转换为 Tensor
        x_tensor = torch.from_numpy(x_np).float()
        y_tensor = torch.tensor(y_val, dtype=torch.long)

        # 应用额外的 transform (如果有)
        if self.transform:
            x_tensor = self.transform(x_tensor)

        # 将离散的标签转换为one-hot向量
        y_tensor = F.one_hot(torch.tensor(y_tensor), num_classes=2)

        return x_tensor, y_tensor, y_tensor

    def _load_all_data(self, flag, cls_mode = -1, modality = "Vibration", motor_id = "all", down_sampling_scale = 1, root_path = "/inspire/hdd/project/continuinglearinginlm/lijiapeng-CZXS25110021/bas_pump/data/NLN-EMP/Dataset"):
        """内部函数：遍历文件并合并通道
            参数:
        - cls_mode: 分类模式，-1表示二分类，其他值表示多分类
        - root_path: 数据集根目录路径
        - modality: 'Vibration' (振动) 或 'Electric' (电流/电压)
        - motor_id: 'Motor-2' 或 'Motor-4'
        - down_sampling_scale: 降采样率，1表示不降采样。设为10或100可以快速调试代码
        """

        # 不直接划分，在后面采用scikit-learn库分层抽样划分
        # if self.flag == 'pretrain':
        #     pass
        # # 需要根据 train 和 test 修改 start_percentage 和 end_percentage
        # elif self.flag == 'train':
        #     # if self.start_percentage is None:
        #     self.start_percentage = 0.0
        #     self.end_percentage = 0.8
        # elif self.flag == 'test':
        #     self.start_percentage = 0.8
        #     self.end_percentage = 1.0
        # print(f'start_percentage: {self.start_percentage}, end_percentage: {self.end_percentage}')



        # 构建搜索路径：root/Modality/Motor/Speed/Label/*.csv
        # 例如: ./Vibration/Motor-2/100/bearing bpfi 1/*.csv
        motor_id = "*" if motor_id == "all" else motor_id
        print("     root_path:"+root_path)
        print("     modality:"+modality)
        print("     motor_id:"+motor_id)
        search_pattern = os.path.join(root_path, modality, motor_id, '*', '*', '*.csv')
        files = glob.glob(search_pattern)
        if not files:
            raise FileNotFoundError(f"在 {root_path} 下未找到文件，请检查路径。")

        # 整理文件组
        experiments = {}
        for file_path in files:
            parent_dir = os.path.dirname(file_path)
            label_name = os.path.basename(parent_dir)
            label_name = re.sub(r'\d+$', '', label_name).strip()



            if label_name == "new motor":
                label_name = "healthy"
            if label_name == "coupling 2D":
                label_name = "coupling"

            # /*TODO*/  标签类别划分合并
            if cls_mode == -1 or cls_mode == 0:
                # 过滤，归类数据，如healthy noise /*TODO*/
                if label_name == "healthy noise":
                    label_name = "healthy"
            elif cls_mode == 1:
                if label_name == "healthy noise":
                    continue
            elif cls_mode == 3:
                if label_name == "healthy noise":
                    continue
                else:
                    label_name = label_name.split()[0]
            elif cls_mode == 11:
                if label_name not in ["healthy", "bearing bpfi", "bearing bpfo", "bearing bsf", "bearing pump", "impeller"]:
                    continue
            elif cls_mode == 12:
                if label_name not in ["healthy", "bearing bpfi", "bearing bpfo", "bearing pump", "soft foot", "stator short", "impeller"]:
                    continue
            elif cls_mode == 13:
                if label_name not in ["healthy", "bearing bpfi", "bearing bpfo", "bearing bsf", "bearing contaminated", "bearing pump", "impeller", "broken rotor bar"]:
                    continue

            speed_dir = os.path.basename(os.path.dirname(parent_dir))
            motor_id = os.path.basename(os.path.dirname(os.path.dirname(parent_dir)))
            filename = os.path.basename(file_path)
            
            try:
                base_name = filename.split('-ch')[0]
                channel_id = int(filename.split('-ch')[1].split('.csv')[0])
            except (IndexError, ValueError):
                continue

            unique_key = f"{motor_id}_{speed_dir}_{label_name}_{base_name}"
            if unique_key not in experiments:
                experiments[unique_key] = {'label': label_name, 'files': {}}
            experiments[unique_key]['files'][channel_id] = file_path

        X_list = []
        y_list = []
        
        # 标签映射
        # 健康映射为0，其余从1开始编号
        # /*TODO*/  标签类别划分合并
        self.class_names = [exp['label'] for exp in experiments.values()]

        # demo test
        # experiments = dict(list(experiments.items())[:1])
        print(f"发现 {len(experiments)} 组实验配置，实验配置统计信息：")
        # print 各类别名称，以及文件夹数目信息
        count_category_occurrences(self.class_names)
        print("开始读取...")

        all_labels_set = set(self.class_names)
        all_labels_set.remove("healthy")
        sorted_abnormal_labels = sorted(list(all_labels_set))
        label_map = {label: i+1 for i, label in enumerate(sorted_abnormal_labels)}
        label_map["healthy"] = 0

        
        for key, exp_data in tqdm(experiments.items(), desc="Loading CSVs"): # key 是 unique_key
            label_idx = label_map[exp_data['label']]
            file_dict = exp_data['files']
            
            num_channels = 5 if modality == 'Vibration' else 6
            if len(file_dict) !=  num_channels: continue # 跳过不完整的通道
            
            channels_data = []
            try:
                # 读取 ch1-ch6
                for ch in range(1, num_channels+1):
                    # 使用 pandas 只读取值，忽略表头，加速读取
                    # 注意：这里假设第一列是 time，后面是 trials
                    df = pd.read_csv(file_dict[ch])
                    if 'time' in df.columns:
                        df = df.drop(columns=['time'])
                    
                    # 在函数slice_sequences_from_list中单独处理降采样，保留降采样的每一个样本，~~no降采样并存入列表~~
                    # channels_data.append(df.values[::down_sampling_scale, :])
                    channels_data.append(df.values)
                
                n_trials = channels_data[0].shape[1]
                # channels_data [C][T, n_trials]

                ###### one-shot
                # if flag == "TRAIN":
                #     # 以每种情况的第一列（第一个trial数据）训
                #     sample = np.stack([ch[:, 0] for ch in channels_data], axis=-1)
                #     X_list.append(sample)
                #     y_list.append(label_idx)

                # elif flag == "TEST":
                #     # 切分 Trials (CSV 的列)
                #     for trial_idx in range(n_trials):
                #         if trial_idx == 0: continue
                #         # 堆叠: np.stack([C][T], axis=-1) => (T,C)
                #         sample = np.stack([ch[:, trial_idx] for ch in channels_data], axis=-1)
                #         X_list.append(sample)
                #         y_list.append(label_idx)
                # else:                
                ###### 

                if self.data_demo and flag == "train":
                    # Demo: only take the middle trial once
                    trial_idx = n_trials // 2
                    sample = np.stack([ch[:, trial_idx] for ch in channels_data], axis=-1)
                    X_list.append(sample)
                    y_list.append(label_idx)

                elif self.project_name == "few_shot": 
                ##### five-shot
                    num4train = min(5, n_trials*4//5)
                    if flag == "train":
                        print("此为训练数据")
                        # 以每种情况的前num4train列（前num4train个trial数据）训
                        for trial_idx in range(num4train):
                            # 堆叠: np.stack([C][T], axis=-1) => (T,C)
                            sample = np.stack([ch[:, trial_idx] for ch in channels_data], axis=-1)
                            X_list.append(sample)
                            y_list.append(label_idx)

                    elif flag == "test":
                        print("此为测试数据")
                        # 切分 Trials (CSV 的列)
                        for trial_idx in range(num4train, n_trials):
                            # 堆叠: np.stack([C][T], axis=-1) => (T,C)
                            sample = np.stack([ch[:, trial_idx] for ch in channels_data], axis=-1)
                            X_list.append(sample)
                            y_list.append(label_idx)
                                
                ##### 
                else:
                # 切分 Trials (CSV 的列)
                    for trial_idx in range(n_trials):
                        # 堆叠: np.stack([C][T], axis=-1) => (T,C)
                        sample = np.stack([ch[:, trial_idx] for ch in channels_data], axis=-1)
                        X_list.append(sample)
                        y_list.append(label_idx)
                    
            except Exception as e:
                print(f"Error reading {key}: {e}")
                continue
                
        # 采用scikit-learn库分层抽样划分
        X_train, X_test, y_train, y_test = train_test_split(
            X_list, y_list,
            test_size=0.2,       # 测试集占20%
            random_state=42,     # 设置随机种子以保证结果可复现
            stratify=y_list      # 关键参数：按照y_list的类别比例进行划分
        )
        
        if self.project_name == "few_shot":
            print("进行few_shot，每个类别训练few组数据，其余进行测试")
        else:
            print(f"一共有 {len(X_list)} 组数据")
            print("进行训练测试集划分...")
            if self.flag == 'train':
                print("此为训练数据")
                X_list=X_train
                y_list=y_train
            elif self.flag == 'test':
                print("此为测试数据，实验setting不再以8:2进行数据划分，测试集即为zeroshot全集")
                # X_list=X_test
                # y_list=y_test
            else:
                pass
        
        # 二分类
        # 在分层抽样划分后进行，类别置 0 (healthy) 和 1 (faulty)
        if cls_mode == -1:
            for i in range(len(y_list)):
                if y_list[i] != 0:
                    y_list[i] = 1

        print("划分后,一共有 {len(X_list)} 组数据,数据样本统计信息：")

        

        # print 各类别数目信息
        count_category_occurrences(y_list)


        X_list, y_list = self.slice_sequences_from_list(
                x_list=X_list,
                y_list=y_list,
                seq_len=self.seq_len,           
                stride_len=self.stride_len,     
                down_sampling_scale=getattr(self, 'down_sampling_scale', 1), # 默认为1
        )

        if len(X_list) == 0:
            print("warning: 数据为空！")


        print(f"切片后，一共有 {len(X_list)} 组数据")

        self.max_seq_len = X_list[0].shape[0]


        # print(f"X_list[0].shape:{X_list[0].shape}")
        # print(f"length:{len(X_list)},{len(y_list)}")
        # print(f"x[0]:{X_list[0]}")
        # print(f"y:{y_list}")

        
        if len(X_list) == 0:
            print("warning: 数据为空！")
            return np.array([]), np.array([]), label_map
            
        return np.array(X_list, dtype=np.float32), np.array(y_list), label_map


    def slice_sequences_from_list(
            self,
            x_list: list, 
            y_list: list, 
            seq_len: int, 
            stride_len: int, 
            down_sampling_scale: int = 1,
            start_percentage: float = 0.0,
            end_percentage: float = 1.0
            ):
        """
        独立于类的序列切片函数，逻辑对应伪代码中的 __read_data__ 和 __getitem__。

        Args:
            x_list: 包含原始时间序列数据的列表 [Array(T, C), ...]
            y_list: 对应的标签列表
            seq_len: 切片窗口长度
            stride_len: 滑动步长
            down_sampling_scale: 降采样倍率 (默认为1)
            start_percentage: 数据截取的起始百分比 (0.0 - 1.0)
            end_percentage: 数据截取的结束百分比 (0.0 - 1.0)

        Returns:
            sliced_x_list: 切片后的数据列表
            sliced_y_list: 切片后对应的标签列表
        """
        sliced_x_list = []
        sliced_y_list = []

        # 对应伪代码: for file in self.file_list: (这里遍历的是已经加载进内存的x_list)
        for idx, data_raw in enumerate(x_list):
            current_label = y_list[idx]

            # 对应伪代码: data = self.__file[file][:]
            # 对应伪代码: for i in range(self.down_sampling_scale):
            for i in range(down_sampling_scale):
                # 对应伪代码: data_i = data[i::self.down_sampling_scale]
                data_i = data_raw[i::down_sampling_scale]

                # file_length为序列总长度total_length
                file_length = data_i.shape[0]

                # 对应伪代码: total_sample_num = (file_length - self.seq_len) // self.stride_len + 1
                if file_length < seq_len:
                    continue # 数据太短无法切片，跳过

                total_sample_num = (file_length - seq_len) // stride_len + 1

                if total_sample_num <= 0:
                    continue

                # Demo模式：只取最中间的一个切片
                if getattr(self, "data_demo", False):
                    mid_start = max(0, (file_length - seq_len) // 2)
                    if mid_start + seq_len <= file_length:
                        segment = data_i[mid_start: mid_start + seq_len]
                        sliced_x_list.append(segment)
                        sliced_y_list.append(current_label)
                    continue

                # 对应伪代码: start_idx = int(total_sample_num * self.start_percentage) * self.stride_len
                # 注意：这里的 idx 是相对于 data_i 的索引
                start_idx = int(total_sample_num * start_percentage) * stride_len

                # 对应伪代码: end_idx = int(total_sample_num * self.end_percentage-1) * self.stride_len
                # 解释：计算最后一个合法的起始索引位置
                end_idx = int(total_sample_num * end_percentage - 1) * stride_len

                # 模拟伪代码中 __getitem__ 的逻辑：
                # 伪代码通过计算 global_index 来定位，这里我们直接生成区间内的所有切片
                # 循环范围从 start_idx 到 end_idx (包含)，步长为 stride_len

                # 这里的 range 必须包含 end_idx，所以是 end_idx + 1
                # 但 range 的步长需要配合 stride_len
                # 另一种写法是 while 循环，更直观匹配逻辑

                curr_start = start_idx
                while curr_start <= end_idx:
                    # 对应伪代码: data = ... [item_start_idx : item_start_idx + self.seq_len]
                    # 边界安全检查
                    if curr_start + seq_len > file_length:
                        break

                    segment = data_i[curr_start : curr_start + seq_len]

                    sliced_x_list.append(segment)
                    sliced_y_list.append(current_label)

                    curr_start += stride_len

        return sliced_x_list, sliced_y_list





    def get_label_map(self):
        """返回标签ID到名称的映射字典"""
        # 反转字典: {0: 'Normal', 1: 'Fault1'...}
        return {v: k for k, v in self.label_map.items()}

def count_category_occurrences(y_list):
    """
    统计 y_list 中所有类别的出现次数，并打印结果。

    Args:
        y_list (list): 包含类别标签的列表。
                       类别可以是数字、字符串或其他可哈希的对象。
    """
    if not y_list:
        print("y_list 为空，无法统计。")
        return

    # 使用 collections.Counter 可以非常高效地统计列表中元素的出现次数
    category_counts = Counter(y_list)
    print("======================================")
    print("类别统计结果:----------")

    # 按照类别名称（或值）排序打印，使输出更整洁
    # 注意：如果类别是混合类型（例如数字和字符串），排序可能会产生警告或意外结果。
    # 如果类别是纯数字，会按数字大小排序。
    # 如果类别是纯字符串，会按字母顺序排序。
    # 为了避免潜在的排序问题，如果您不关心打印顺序，可以直接遍历 category_counts.items()
    sorted_categories = sorted(category_counts.keys())

    for category in sorted_categories:
        count = category_counts[category]
        # 使用 f-string 格式化输出，使其更易读
        print(f"类别 '{category}': {count} 次")

    print("--------------------")
    print(f"总类别数: {len(sorted_categories)}")
    print(f"总数: {len(y_list)}")
    print("======================================")



class NLNEMP_ElecLoader(NLNEMPloader):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, down_sampling_scale=1, label_type='local', flag=None, motor_id='all', normalize=True, transform=None):
        super().__init__(
            root_path=root_path,
            args=args,
            seq_len=seq_len,
            stride_len=stride_len,
            down_sampling_scale=down_sampling_scale,
            label_type=label_type,
            flag=flag,
            modality="Electric",
            motor_id=motor_id,
            normalize=normalize,
            transform=transform,
        )


class NLNEMP_VibLoader(NLNEMPloader):
    def __init__(self, root_path, args, seq_len=1024, stride_len=1024, down_sampling_scale=1, label_type='local', flag=None, motor_id='all', normalize=True, transform=None):
        super().__init__(
            root_path=root_path,
            args=args,
            seq_len=seq_len,
            stride_len=stride_len,
            down_sampling_scale=down_sampling_scale,
            label_type=label_type,
            flag=flag,
            modality="Vibration",
            motor_id=motor_id,
            normalize=normalize,
            transform=transform,
        )


# from data_provider.uea import subsample, interpolate_missing, Normalizer
class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        # NumPy 路径：对序列和通道分开处理，按通道标准化，避免不同通道量纲被混合
        if isinstance(df, np.ndarray):
            print("执行按通道标准化")
            eps = np.finfo(float).eps
            # 期望形状 [N, T, C] 或 [N, C]
            if df.ndim == 3:
                # 在样本和时间两个维度上求每个通道的统计量
                axes = (0, 1)
            elif df.ndim == 2:
                axes = (0,)
            else:
                raise ValueError(f"Unsupported ndarray shape for normalization: {df.shape}")

            if self.norm_type == "standardization":
                if self.mean is None:
                    self.mean = df.mean(axis=axes)
                    self.std = df.std(axis=axes)
                # 维度对齐以便广播
                mean = self.mean.reshape((1,) * (df.ndim - 1) + (-1,))
                std = self.std.reshape((1,) * (df.ndim - 1) + (-1,))
                return (df - mean) / (std + eps)

            elif self.norm_type == "minmax":
                if self.max_val is None:
                    self.max_val = df.max(axis=axes)
                    self.min_val = df.min(axis=axes)
                min_val = self.min_val.reshape((1,) * (df.ndim - 1) + (-1,))
                max_val = self.max_val.reshape((1,) * (df.ndim - 1) + (-1,))
                return (df - min_val) / (max_val - min_val + eps)

            else:
                raise (NameError(f'Normalize method "{self.norm_type}" not implemented for ndarray input'))

        print("执行普通的全部数据标准化")
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))
