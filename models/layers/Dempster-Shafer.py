import numpy as np

class DSTFewShotClassifier:
    def __init__(self, threshold=0.5):
        """
        确信理论少样本分类器。

        参数:
        threshold: float，用于决定某个类别是否可信的信任度阈值
        """
        self.threshold = threshold
        self.prototypes = {}

    def fit(self, features):
        """
        为每个类别计算原型向量，作为少样本确信分类的基础。

        参数:
        features: dict，每个类别对应的样本特征列表，键为类别ID
        """
        for label, vectors in features.items():
            self.prototypes[label] = np.mean(vectors, axis=0)  # 取均值作为原型

    def compute_bpa(self, sample, prototype):
        """
        计算基本概率赋值（BPA）。

        参数:
        sample: numpy数组，表示新样本的特征向量
        prototype: numpy数组，类别的原型向量

        返回:
        BPA值，表示样本对该类别的支持度
        """
        distance = np.linalg.norm(sample - prototype)
        bpa = np.exp(-distance)  # 使用距离的负指数函数表示支持度
        return bpa

    def predict(self, sample):
        """
        对新样本进行预测并量化不确定性。

        参数:
        sample: numpy数组，表示新样本的特征向量

        返回:
        predictions: 预测类别
        belief_scores: 每个类别的信任度
        plausibility_scores: 每个类别的似然度
        """
        bpa_values = {label: self.compute_bpa(sample, prototype) for label, prototype in self.prototypes.items()}
        total_bpa = sum(bpa_values.values())

        # 计算信任度（Belief）和似然度（Plausibility）
        belief_scores = {label: bpa / total_bpa for label, bpa in bpa_values.items()}
        plausibility_scores = {label: 1 - (1 - bpa / total_bpa) for label, bpa in bpa_values.items()}

        # 使用最大信任度的类别作为预测类别
        predictions = max(belief_scores, key=belief_scores.get) if max(belief_scores.values()) >= self.threshold else "Uncertain"
        
        return predictions, belief_scores, plausibility_scores

# 使用示例
if __name__ == "__main__":
    # 假设每个类别有少量样本特征
    features = {
        'class_1': [np.random.rand(10) for _ in range(5)],
        'class_2': [np.random.rand(10) for _ in range(5)],
        'class_3': [np.random.rand(10) for _ in range(5)]
    }

    # 初始化分类器
    classifier = DSTFewShotClassifier(threshold=0.5)
    classifier.fit(features)

    # 新样本预测
    new_sample = np.random.rand(10)
    prediction, belief_scores, plausibility_scores = classifier.predict(new_sample)
    print("预测类别:", prediction)
    print("信任度分数:", belief_scores)
    print("似然度分数:", plausibility_scores)
