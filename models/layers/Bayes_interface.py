import numpy as np
from scipy.stats import multivariate_normal
from sklearn.preprocessing import normalize

class BayesianFewShotClassifier:
    def __init__(self, sparsity_level=0.1, num_samples=100, noise_scale=0.05):
        """
        初始化贝叶斯少样本分类器，支持每类别只有一个数据的情况。

        参数:
        sparsity_level: float，控制稀疏度的参数
        num_samples: int，从后验分布中采样的次数
        noise_scale: float，单样本伪数据生成时的噪声标准差
        """
        self.sparsity_level = sparsity_level
        self.num_samples = num_samples
        self.noise_scale = noise_scale
        self.prototypes = {}
        self.uncertainties = {}

    def fit(self, features):
        """
        根据少样本数据构建稀疏原型向量和不确定性量化。

        参数:
        features: dict，每个类别对应的样本特征列表，键为类别ID
        """
        for label, vectors in features.items():
            # Step 1: 处理每类只有一个样本的情况，生成伪样本
            if len(vectors) == 1:
                # 生成多个伪样本
                vector = vectors[0]
                pseudo_samples = [vector + np.random.normal(0, self.noise_scale, len(vector)) for _ in range(self.num_samples)]
                pseudo_samples = np.array(pseudo_samples)
            else:
                pseudo_samples = np.array(vectors)

            # Step 2: 使用伪样本的均值和协方差作为估计
            sample_mean = np.mean(pseudo_samples, axis=0)
            sample_cov = np.cov(pseudo_samples.T) + 1e-6 * np.eye(len(sample_mean))  # 确保协方差正定

            # Step 3: 定义强先验并进行贝叶斯更新
            mean_prior = np.zeros(len(sample_mean))
            cov_prior = np.eye(len(sample_mean))  # 可以根据先验知识设定更合适的协方差

            cov_post = np.linalg.inv(np.linalg.inv(cov_prior) + np.linalg.inv(sample_cov))
            mean_post = cov_post @ (np.linalg.inv(cov_prior) @ mean_prior + np.linalg.inv(sample_cov) @ sample_mean)

            # Step 4: 从后验中采样并稀疏化
            samples = multivariate_normal.rvs(mean=mean_post, cov=cov_post, size=self.num_samples)
            sparse_prototype = np.mean(samples, axis=0)
            sparse_prototype[np.abs(sparse_prototype) < self.sparsity_level * np.max(np.abs(sparse_prototype))] = 0  # 稀疏化

            # 计算原型不确定性（使用后验协方差的对角元素）
            uncertainty = np.diag(cov_post)

            self.prototypes[label] = sparse_prototype
            self.uncertainties[label] = uncertainty

    def predict(self, sample):
        """
        对新样本进行预测并量化不确定性。

        参数:
        sample: numpy数组，表示新样本的特征向量

        返回:
        predictions: 预测类别
        uncertainty_scores: 每个类别的预测不确定性分数
        """
        distances = {}
        uncertainty_scores = {}

        for label, prototype in self.prototypes.items():
            # Step 1: 计算样本到类别原型的欧式距离
            distance = np.linalg.norm(sample - prototype)

            # Step 2: 结合类别不确定性进行置信度度量
            uncertainty = np.sum(self.uncertainties[label]) + 0.1  # 添加偏置，反映单样本高不确定性
            adjusted_distance = distance * (1 + uncertainty)

            distances[label] = adjusted_distance
            uncertainty_scores[label] = uncertainty

        # Step 3: 根据最小修正距离进行分类
        predictions = min(distances, key=distances.get)

        return predictions, uncertainty_scores

# 使用示例
if __name__ == "__main__":
    # 假设每个类别只有一个样本特征
    features = {
        'class_1': [np.random.rand(10)],
        'class_2': [np.random.rand(10)],
        'class_3': [np.random.rand(10)]
    }

    # 初始化分类器
    classifier = BayesianFewShotClassifier(sparsity_level=0.1, num_samples=100, noise_scale=0.05)
    classifier.fit(features)

    # 新样本预测
    new_sample = np.random.rand(10)
    prediction, uncertainty_scores = classifier.predict(new_sample)
    print("预测类别:", prediction)
    print("不确定性分数:", uncertainty_scores)
