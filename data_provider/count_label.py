import h5py

# 打开HDF5文件
with h5py.File('/dataWX/WYL/PHM-Large-Model/PHM_dataset/XJTU_CLS/XJTU_CLS.hdf5', 'r') as file:
    # 创建一个空集合来存储唯一的label值
    unique_labels = set()

    # 遍历文件中的所有数据集
    for dataset_name in file:
        # 确保当前对象是数据集
        if isinstance(file[dataset_name], h5py.Dataset):
            # 获取label属性值
            label = file[dataset_name].attrs.get('label')
            if label is not None:
                unique_labels.add(label)

    # 打印出唯一的label值及其数量
    print(f"共有 {len(unique_labels)} 种不同的label:")
    for label in unique_labels:
        print(label)