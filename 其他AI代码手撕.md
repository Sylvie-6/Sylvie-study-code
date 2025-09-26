# Softmax
```python
import numpy as np

def softmax(x):
    """
    计算softmax函数
    
    参数:
    x -- 输入数组或矩阵，形状为(样本数, 特征数)或(特征数,)
    
    返回:
    与输入形状相同的数组，其中每个元素表示对应位置的softmax值
    """
    # 处理单样本情况（一维数组）
    if x.ndim == 1:
        # 减去最大值以提高数值稳定性
        x_max = np.max(x)
        x_exp = np.exp(x - x_max)
        return x_exp / np.sum(x_exp)
    else:
        # 处理多样本情况（二维数组）
        # 沿特征轴寻找最大值
        x_max = np.max(x, axis=1, keepdims=True)
        # 减去最大值并计算指数
        x_exp = np.exp(x - x_max)
        # 计算每行的和并归一化
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

# 测试代码
if __name__ == "__main__":
    # 测试单个向量
    x = np.array([1.0, 2.0, 3.0])
    print("单个向量的softmax结果:")
    print(softmax(x))
    print("和为:", np.sum(softmax(x)))  # 应该接近1
    
    # 测试矩阵（多个样本）
    x = np.array([[1.0, 2.0, 3.0], 
                  [4.0, 5.0, 6.0]])
    print("\n矩阵的softmax结果:")
    print(softmax(x))
    print("每行的和:")
    print(np.sum(softmax(x), axis=1))  # 每行的和都应该接近1
```
---
# 交叉熵损失函数
```python
import torch
import torch.nn.functional as F

# 定义 softmax 函数（手动实现，也可直接用 F.softmax）
def softmax(scores):
    """将 logits 通过 softmax 转换为概率分布，保证数值稳定性"""
    # scores: 形状为 (样本数, 类别数) 的 torch.Tensor
    scores_max = torch.max(scores, dim=1, keepdim=True).values  # 按行取最大值（保持维度便于广播）
    scores_exp = torch.exp(scores - scores_max)  # 减去最大值防止指数溢出
    return scores_exp / torch.sum(scores_exp, dim=1, keepdim=True)  # 按行归一化

# 定义交叉熵损失函数
def cross_entropy(pred_scores, true_labels):
    """手动计算多分类交叉熵损失，包含数值稳定性裁剪"""
    # pred_scores: 模型输出的 logits，形状 (batch_size, num_classes)
    # true_labels: 真实类别标签，形状 (batch_size,)，需为 torch.LongTensor 类型
    probs_matrix = softmax(pred_scores)  # 转换为概率分布
    n_samples = true_labels.shape[0]     # 样本数量
    
    # 限制概率范围，避免 log(0) 或 log(1) 导致数值问题
    eps = 1e-12
    probs_matrix = torch.clamp(probs_matrix, eps, 1.0 - eps)
    
    # 提取每个样本“真实标签对应的预测概率”
    correct_class_probs = probs_matrix[torch.arange(n_samples), true_labels]
    
    # 计算负对数似然并求平均
    log_likelihood = -torch.log(correct_class_probs)
    return torch.sum(log_likelihood) / n_samples

# ------------------- 示例：生成数据并测试 ------------------- #
torch.manual_seed(42)  # 固定随机种子以保证结果可复现

# 模拟模型输出：5 个样本，每个样本有 4 个类别得分（logits）
sample_logits = torch.randn(5, 4)  

# 模拟真实标签：5 个样本的类别（取值 0~3）
sample_labels = torch.randint(0, 4, size=(5,))  

# 计算交叉熵损失
loss = cross_entropy(sample_logits, sample_labels)
print("交叉熵损失：", loss.item())
```
```python
import torch
import torch.nn.functional as F

torch.manual_seed(42)
sample_logits = torch.randn(5, 4)  
sample_labels = torch.randint(0, 4, size=(5,))  

# 直接用 PyTorch 内置交叉熵函数（推荐，数值稳定性更好）
loss = F.cross_entropy(sample_logits, sample_labels)
print("内置交叉熵损失：", loss.item())
```
---
# KNN
```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        """初始化KNN分类器（默认k=3）"""
        self.k = k
        self.X_train = None  # 训练样本特征
        self.y_train = None  # 训练样本标签
    
    def fit(self, X, y):
        """训练：KNN为“惰性学习”，仅存储训练数据"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_test):
        """预测：对每个测试样本，找k个最近邻并投票"""
        predictions = []
        for x in X_test:
            # 1. 计算测试样本与所有训练样本的**欧氏距离**
            distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            
            # 2. 按距离排序，取前k个样本的**索引**
            k_indices = np.argsort(distances)[:self.k]
            
            # 3. 提取k个样本的**标签**
            k_labels = self.y_train[k_indices]
            
            # 4. 多数投票（统计出现次数最多的标签）
            most_common = Counter(k_labels).most_common(1)[0][0]
            predictions.append(most_common)
        
        return np.array(predictions)


# ------------------- 测试用例（面试时可快速演示） ------------------- #
if __name__ == "__main__":
    # 模拟训练数据：4个样本，2个特征，标签为0/1
    X_train = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
    y_train = np.array([0, 0, 0, 1])
    
    # 模拟测试数据：2个待预测样本
    X_test = np.array([[4, 5], [7, 8]])
    
    # 初始化并“训练”KNN
    knn = KNN(k=3)
    knn.fit(X_train, y_train)
    
    # 预测并打印结果
    y_pred = knn.predict(X_test)
    print("测试样本的预测类别：", y_pred)
```
---
# K-Means
```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, tol=1e-4):
        """
        初始化KMeans聚类器
        :param n_clusters: 聚类数量(k)
        :param max_iter: 最大迭代次数
        :param tol: 质心变化阈值，小于此值认为收敛
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None  # 质心数组

    def fit(self, X):
        """
        训练KMeans模型
        :param X: 输入数据，形状为(n_samples, n_features)
        """
        # 1. 初始化质心：从样本中随机选择k个作为初始质心
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # 2. 分配样本：计算每个样本到质心的距离，分配到最近的簇
            # 计算距离（欧氏距离的平方，避免开方运算，结果等价）
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)  # 每个样本所属簇的索引
            
            # 3. 更新质心：计算每个簇的均值作为新质心
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            
            # 4. 检查收敛：质心变化小于阈值则停止迭代
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            
            self.centroids = new_centroids

    def predict(self, X):
        """
        预测样本所属簇
        :param X: 输入数据，形状为(n_samples, n_features)
        :return: 每个样本的簇标签
        """
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)


# 测试代码（面试时可快速演示）
if __name__ == "__main__":
    # 生成模拟数据（3个簇）
    X = np.vstack([
        np.random.normal(0, 0.5, size=(100, 2)),
        np.random.normal(5, 0.5, size=(100, 2)),
        np.random.normal(10, 0.5, size=(100, 2))
    ])
    
    # 聚类
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    labels = kmeans.predict(X)
    
    print("质心坐标：")
    print(kmeans.centroids)
    print("\n前10个样本的簇标签：")
    print(labels[:10])
```
