
import torch
import numpy as np

class BalancedDataLoaderIterator:
    def __init__(self, dataloaders):
        self.dataloaders = dataloaders
        self.num_dataloaders = len(dataloaders)

        # 计算每个数据加载器的长度
        lengths = torch.tensor([len(dataloader) for dataloader in dataloaders], dtype=torch.float)
        
        # # 使用 Log 采样权重，确保 lengths 是浮点数计算 Log
        # weights = 1.0 / torch.log(lengths.float() + 1)
        # self.probabilities = weights / weights.sum()

        # 根据长度采样
        # self.probabilities = lengths / lengths.sum()
        # 均匀采样
        self.probabilities = torch.ones(
            self.num_dataloaders, dtype=torch.float) / self.num_dataloaders
        self.total_length = int(lengths.sum())
        self.current_iteration = 0

        print("Data loader lengths:", lengths)
        print("Sampling probabilities:", self.probabilities)
        print("Total length:", self.total_length)

    def __iter__(self):
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]
        self.current_iteration = 0
        return self

    def __next__(self):
        if self.current_iteration >= self.total_length:
            raise StopIteration

        # 根据动态概率选择一个数据加载器
        chosen_index = torch.multinomial(self.probabilities, 1).item()

        try:
            # 从被选中的数据加载器中获取一个批次
            sample = next(self.iterators[chosen_index])
        except StopIteration:
            # 如果数据加载器数据耗尽，用伪样本填充
            sample = self.generate_fake_samples_for_batch(chosen_index, batch_size=1)  # 设定所需批次大小
            # 重新初始化该数据加载器的迭代器，若后续需要
            self.iterators[chosen_index] = iter(self.dataloaders[chosen_index])
        self.current_iteration += 1
        return sample, chosen_index

    def __len__(self):
        return self.total_length

    def generate_fake_samples_for_batch(self, dataloader_id, batch_size):
        """
        从指定数据集中随机抽取一个真实样本，而不是生成全零样本
        """
        # 验证dataloader_id的有效性
        if dataloader_id >= len(self.dataloaders) or dataloader_id < 0:
            raise ValueError("Invalid dataloader ID")

        # 获取数据集
        dataloader = self.dataloaders[dataloader_id]
        dataset = dataloader.dataset
        
        # 从数据集中随机抽取一个样本
        try:
            # 生成一个随机索引
            random_index = torch.randint(0, len(dataset), (1,)).item()
            
            # 获取随机样本
            sample = dataset[random_index]
            
            # 如果样本不是列表或元组形式，转换为列表便于处理
            if not isinstance(sample, (list, tuple)):
                sample = [sample]
                
            # 转换为批次形式（如果需要）
            samples = []
            for s in sample:
                if isinstance(s, torch.Tensor):
                    s = s.unsqueeze(0).repeat(batch_size, *(1 for _ in range(s.dim())))
                    samples.append(s)
                else:
                    samples.append(s)
            
            return samples
            
        except Exception as e:
            print(f"Error sampling from dataset {dataloader_id}: {e}")
            # 如果随机抽取失败，回退到原来的全零样本方法
            # 重新初始化一个迭代器用于获取样本形状
            iterator = iter(dataloader)
            try:
                sample_batch = next(iterator)
                fake_samples = []

                # 为每个样本维度创建全零张量的伪样本
                for sample in sample_batch:
                    if isinstance(sample, torch.Tensor):
                        fake_sample = torch.zeros([batch_size] + list(sample.shape)[1:])
                        fake_samples.append(fake_sample)
                    else:
                        # 可以根据需要处理其他数据类型
                        fake_samples.append(sample)

                return fake_samples
            except StopIteration:
                print(f"Failed to create fake sample for dataloader {dataloader_id}")
                return None
