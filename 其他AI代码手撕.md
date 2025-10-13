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
import numpy as np

def cross_entropy_loss(y_true, y_pred_logits):
    """
    计算多分类交叉熵损失（输入为模型输出的logits，未经过Softmax）
    :param y_true: 真实标签，形状为[样本数, 类别数]，one-hot编码（如[0,1,0]）
    :param y_pred_logits: 模型原始输出（logits），形状为[样本数, 类别数]
    :return: 平均交叉熵损失（标量）
    """
    # 步骤1：Softmax计算（加最大值做数值稳定）
    exp_logits = np.exp(y_pred_logits - np.max(y_pred_logits, axis=1, keepdims=True))
    y_pred_softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # 步骤2：计算交叉熵（加epsilon避免log(0)）
    epsilon = 1e-10  # 极小值，防止数值错误
    cross_entropy = -np.sum(y_true * np.log(y_pred_softmax + epsilon), axis=1)
    
    # 步骤3：返回平均损失（对所有样本求平均）
    return np.mean(cross_entropy)

# 测试代码
if __name__ == "__main__":
    # 模拟1个样本，3分类任务：真实标签为类别1（one-hot），模型logits输出
    y_true = np.array([[0, 1, 0]])  # 真实标签（one-hot）
    y_pred_logits = np.array([[2.0, 5.0, 1.0]])  # 模型原始输出（未Softmax）
    
    loss = cross_entropy_loss(y_true, y_pred_logits)
    print("平均交叉熵损失:", loss)  # 输出约0.0189（符合预期，模型对正确类别预测置信度高）
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
时间复杂度：`O(n×m)`
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
            most_common = Counter(k_labels).most_common(1)[0][0]  #.most_common(1)1：这是 Counter 对象的一个方法，用于返回出现次数最多的元素及其计数
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
        self.centroids = None  # 对 “质心数组” 的初始化占位

    def fit(self, X):
        """
        训练KMeans模型
        :param X: 输入数据，形状为(n_samples, n_features)
        """
        # 1. 初始化质心：从样本中随机选择k个作为初始质心
        np.random.seed(42)  # 固定随机种子，保证结果可复现
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        # np.random.choice(..., ..., ...)：NumPy 的随机选择函数，用于从指定范围中随机挑选元素
        # replace=False：指定 “不允许重复选择”
        
        for _ in range(self.max_iter):
            # 2. 分配样本：计算每个样本到质心的距离，分配到最近的簇
            # 计算距离（欧氏距离的平方，避免开方运算，结果等价）
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            # self.centroids[:, np.newaxis]：扩展质心维度，为广播做准备
            # [:, np.newaxis] 是在数组的第二维度插入一个新轴（增加一个维度），将形状从 (n_clusters, n_features) 变为 (n_clusters, 1, n_features)。（比如插在第三个维度应该是[:, :, np.newaxis]）
            # .sum(axis=2)：对特征维度求和
            
            labels = np.argmin(distances, axis=0)  # 每个样本所属簇的索引 np.argmin 是 NumPy 中用于查找 “最小值索引” 的函数。axis=0 表示 “沿着第一个维度（质心维度）查找最小值”。
            
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
        return np.argmin(distances, axis=0) #沿着“样本数量”这个维度（即每一列） 找最小值的索引


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
---
# 线性回归
```python
import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate    # 学习率
        self.n_iter = n_iterations # 迭代次数
        self.weights = None        # 特征权重
        self.bias = None           # 偏置项

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 初始化参数（权重为0，偏置为0）
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # 前向传播：计算预测值
            y_pred = np.dot(X, self.weights) + self.bias
            # 计算梯度（均方误差对权重、偏置的导数）
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)
            # 梯度下降更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # 用学习到的参数做预测
        return np.dot(X, self.weights) + self.bias


# 测试：生成模拟数据并训练
if __name__ == "__main__":
    np.random.seed(42)
    # 生成特征（100个样本，1个特征）
    X = np.random.rand(100, 1)
    # 真实关系：y = 2x + 3 + 随机噪声
    y = 2 * X.squeeze() + 3 + np.random.randn(100) * 0.1

    model = LinearRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    print("学习到的权重:", model.weights)   # 应接近 [2.]
    print("学习到的偏置:", model.bias)     # 应接近 3.
    
    # 预测示例
    X_test = np.array([[0], [1]])
    print("测试预测值:", model.predict(X_test))
```
---
# 逻辑回归
```python
import numpy as np

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate    # 学习率
        self.n_iter = n_iterations # 迭代次数
        self.weights = None        # 特征权重
        self.bias = None           # 偏置项

    def sigmoid(self, z):
        # sigmoid激活函数：将线性输出映射到(0,1)概率区间
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # 初始化参数（权重为0，偏置为0）
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            # 前向传播：计算正类概率
            z = np.dot(X, self.weights) + self.bias
            y_pred_proba = self.sigmoid(z)
            # 计算梯度（交叉熵损失对权重、偏置的导数）
            dw = (1 / n_samples) * np.dot(X.T, (y_pred_proba - y))
            db = (1 / n_samples) * np.sum(y_pred_proba - y)
            # 梯度下降更新参数
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        # 预测“属于正类”的概率
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        # 根据概率阈值（默认0.5）预测类别（0或1）
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)


# 测试：生成模拟二分类数据并训练
if __name__ == "__main__":
    np.random.seed(42)
    # 生成特征（100个样本，2个特征）
    X = np.random.randn(100, 2)
    # 真实分类规则：y=1 当 2x₁ - 3x₂ + 1 > 0
    y = (2 * X[:, 0] - 3 * X[:, 1] + 1 > 0).astype(int)

    model = LogisticRegressionGD(learning_rate=0.1, n_iterations=1000)
    model.fit(X, y)
    
    # 预测示例
    X_test = np.array([[0, 0], [1, 1]])
    print("正类概率预测:", model.predict_proba(X_test))
    print("类别预测:", model.predict(X_test))
```
---
# 2D卷积
### 包含stride和padding
```python
import numpy as np

def conv2d_full(inputs, kernels, stride=1, padding=0):
    """
    2D卷积完整实现（支持stride和padding）
    :param inputs: 输入特征图，形状为[H, W, C]（高、宽、通道数）
    :param kernels: 卷积核，形状为[K, K, C, N]（核高、核宽、输入通道数、输出通道数）
    :param stride: 步长，默认1（横向/纵向步长相同）
    :param padding: 零填充数，默认0（上下/左右各填充padding层）
    :return: 输出特征图，形状为[H_out, W_out, N]
    """
    # 1. 解析输入和卷积核维度
    H, W, C = inputs.shape  # 输入高、宽、通道数
    K, _, C_kernel, N = kernels.shape  # 核高、输入通道数、输出通道数
    
    # 2. 合法性检查（输入通道数需与卷积核输入通道数一致）
    assert C == C_kernel, f"输入通道数{C}与卷积核输入通道数{C_kernel}不匹配"
    
    # 3. 对输入进行零填充（上下左右各补padding行/列）
    padded_input = np.pad(
        inputs, 
        pad_width=((padding, padding), (padding, padding), (0, 0)),  # (高方向补, 宽方向补, 通道不补)
        mode='constant',  # 填充0
        constant_values=0
    )
    
    # 4. 计算输出特征图尺寸（卷积尺寸公式）
    H_out = (H + 2 * padding - K) // stride + 1  # 输出高
    W_out = (W + 2 * padding - K) // stride + 1  # 输出宽
    
    # 5. 初始化输出特征图（全0）
    output = np.zeros((H_out, W_out, N))
    
    # 6. 滑动窗口计算卷积（遍历输出每个像素、每个输出通道）
    for n in range(N):  # 遍历输出通道（每个通道对应1个卷积核）
        for h in range(H_out):  # 遍历输出高维度
            for w in range(W_out):  # 遍历输出宽维度
                # 计算当前滑动窗口在填充后输入上的起始位置
                h_start = h * stride
                h_end = h_start + K
                w_start = w * stride
                w_end = w_start + K
                
                # 提取输入窗口（[K, K, C]）和当前卷积核（[K, K, C]）
                input_window = padded_input[h_start:h_end, w_start:w_end, :]
                kernel = kernels[:, :, :, n]
                
                # 卷积计算：窗口与核元素相乘 → 所有元素求和 → 存入输出
                output[h, w, n] = np.sum(input_window * kernel)
    
    return output

# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # 模拟输入：28x28灰度图（通道数C=1）
    input_img = np.random.randn(28, 28, 1)  # [28,28,1]
    # 模拟卷积核：3x3大小，输入通道1，输出通道4（4个卷积核）
    kernels = np.random.randn(3, 3, 1, 4)  # [3,3,1,4]
    
    # 测试1：stride=1, padding=0（无填充，步长1）
    output1 = conv2d_full(input_img, kernels, stride=1, padding=0)
    print("测试1输出尺寸（stride=1, padding=0）:", output1.shape)  # 输出 (26,26,4)（28-3+1=26）
    
    # 测试2：stride=2, padding=1（补1层零，步长2）
    output2 = conv2d_full(input_img, kernels, stride=2, padding=1)
    print("测试2输出尺寸（stride=2, padding=1）:", output2.shape)  # 输出 (14,14,4)（(28+2-3)/2+1=14）
```
### 无stride和padding
```python
import numpy as np

def conv2d_simple(inputs, kernels):
    """
    2D卷积简化实现（固定stride=1，padding=0）
    :param inputs: 输入特征图，形状[H, W, C]
    :param kernels: 卷积核，形状[K, K, C, N]
    :return: 输出特征图，形状[H-K+1, W-K+1, N]
    """
    # 解析维度
    H, W, C = inputs.shape
    K, _, C_kernel, N = kernels.shape
    # 计算输出尺寸（无padding、stride=1，直接用输入尺寸减核尺寸加1）
    H_out = H - K + 1
    W_out = W - K + 1
    # 初始化输出
    output = np.zeros((H_out, W_out, N))
    
    # 滑动窗口卷积（核心逻辑与完整版本一致）
    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                # 输入窗口：无需计算stride（固定1），直接从h/w开始取K大小
                input_window = inputs[h:h+K, w:w+K, :]
                kernel = kernels[:, :, :, n]
                output[h, w, n] = np.sum(input_window * kernel)
    
    return output

# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    input_img = np.random.randn(28, 28, 1)  # [28,28,1]
    kernels = np.random.randn(3, 3, 1, 2)  # [3,3,1,2]
    output = conv2d_simple(input_img, kernels)
    print("简化版输出尺寸:", output.shape)  # 输出 (26,26,2)（28-3+1=26）
```
# CNN池化
### 平均池化
```python
import numpy as np

def avg_pool2d(inputs, ksize=2, stride=2):
    """
    2D平均池化实现（通道独立，默认丢弃边界不完整窗口）
    :param inputs: 输入特征图，形状为[H, W, C]（高、宽、通道数）
    :param ksize: 池化核大小，默认2（正方形核，ksize×ksize）
    :param stride: 滑动步长，默认2（横向/纵向步长相同）
    :return: 输出特征图，形状为[H_out, W_out, C]
    """
    # 1. 解析输入维度：H(高)、W(宽)、C(通道数)
    H, W, C = inputs.shape
    
    # 2. 计算输出特征图尺寸（丢弃边界不完整窗口，确保窗口完全在输入内）
    # 公式：输出尺寸 = (输入尺寸 - 池化核大小) // 步长 + 1
    H_out = (H - ksize) // stride + 1
    W_out = (W - ksize) // stride + 1
    
    # 3. 初始化输出特征图（通道数与输入一致，高宽为计算出的H_out/W_out）
    output = np.zeros((H_out, W_out, C), dtype=inputs.dtype)
    
    # 4. 遍历计算：通道独立处理，每个通道单独做平均池化
    for c in range(C):  # 遍历每个通道（池化不跨通道，通道间独立）
        for h in range(H_out):  # 遍历输出的高维度
            for w in range(W_out):  # 遍历输出的宽维度
                # 计算当前池化窗口在输入上的起始/结束坐标
                h_start = h * stride  # 窗口起始高
                h_end = h_start + ksize  # 窗口结束高（左闭右开）
                w_start = w * stride  # 窗口起始宽
                w_end = w_start + ksize  # 窗口结束宽
                
                # 提取当前通道的输入窗口（形状：[ksize, ksize]）
                input_window = inputs[h_start:h_end, w_start:w_end, c]
                
                # 窗口内元素求平均，赋值给输出对应位置
                output[h, w, c] = np.mean(input_window)
    
    return output

# ---------------------- 测试代码 ----------------------
if __name__ == "__main__":
    # 1. 测试1：小尺寸输入（验证基础功能）
    # 输入：4x4x1（高4、宽4、1通道，模拟灰度特征图）
    input_small = np.array([
        [[1], [2], [3], [4]],
        [[5], [6], [7], [8]],
        [[9], [10], [11], [12]],
        [[13], [14], [15], [16]]
    ])
    # 池化：核大小2×2，步长2
    output_small = avg_pool2d(input_small, ksize=2, stride=2)
    print("测试1 - 小尺寸输入输出:")
    print("输入形状:", input_small.shape)  # 输出: (4, 4, 1)
    print("输出形状:", output_small.shape)  # 输出: (2, 2, 1)（(4-2)//2+1=2）
    print("输出值:\n", output_small.squeeze())  #  squeeze()去除通道维度，输出: [[3.5, 5.5], [11.5, 13.5]]

    # 2. 测试2：多通道输入（模拟RGB类特征图）
    # 输入：28x28x3（高28、宽28、3通道）
    input_rgb = np.random.randn(28, 28, 3)
    # 池化：核大小3×3，步长1（无下采样，仅平滑）
    output_rgb = avg_pool2d(input_rgb, ksize=3, stride=1)
    print("\n测试2 - 多通道输入输出:")
    print("输入形状:", input_rgb.shape)  # 输出: (28, 28, 3)
    print("输出形状:", output_rgb.shape)  # 输出: (26, 26, 3)（(28-3)//1+1=26）
```
### 最大池化
将代码中`np.mean(input_window)`替换为
`np.max(input_window)`，逻辑完全一致，仅窗口内聚合方式不同。

---
# 手撕神经网络
## CNN

## GRU
```python
import numpy as np

class GRU:
    def __init__(self, input_dim, hidden_dim):
        """
        初始化GRU模型
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏状态维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # 初始化权重参数（更新门、重置门、候选隐藏状态）
        # 输入权重 (input_dim, hidden_dim)
        self.Wz = np.random.randn(input_dim, hidden_dim) * 0.01  # 更新门输入权重
        self.Wr = np.random.randn(input_dim, hidden_dim) * 0.01  # 重置门输入权重
        self.Wh = np.random.randn(input_dim, hidden_dim) * 0.01  # 候选状态输入权重
        
        # 隐藏状态权重 (hidden_dim, hidden_dim)
        self.Uz = np.random.randn(hidden_dim, hidden_dim) * 0.01  # 更新门隐藏权重
        self.Ur = np.random.randn(hidden_dim, hidden_dim) * 0.01  # 重置门隐藏权重
        self.Uh = np.random.randn(hidden_dim, hidden_dim) * 0.01  # 候选状态隐藏权重
        
        # 偏置 (1, hidden_dim)
        self.bz = np.zeros((1, hidden_dim))  # 更新门偏置
        self.br = np.zeros((1, hidden_dim))  # 重置门偏置
        self.bh = np.zeros((1, hidden_dim))  # 候选状态偏置

    def sigmoid(self, x):
        """sigmoid激活函数（门控输出范围0~1）"""
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev):
        """
        GRU前向传播（单时间步）
        :param x: 当前时间步输入，形状 (1, input_dim)
        :param h_prev: 上一时间步隐藏状态，形状 (1, hidden_dim)
        :return: 当前时间步隐藏状态h
        """
        # 1. 计算更新门 z（控制保留多少历史信息）
        z = self.sigmoid(np.dot(x, self.Wz) + np.dot(h_prev, self.Uz) + self.bz)
        
        # 2. 计算重置门 r（控制如何结合历史信息和当前输入）
        r = self.sigmoid(np.dot(x, self.Wr) + np.dot(h_prev, self.Ur) + self.br)
        
        # 3. 计算候选隐藏状态 h~（基于重置门过滤后的历史信息）
        h_tilde = np.tanh(np.dot(x, self.Wh) + np.dot(r * h_prev, self.Uh) + self.bh)
        
        # 4. 更新隐藏状态 h（z控制历史与候选状态的融合比例）
        h = (1 - z) * h_prev + z * h_tilde
        
        return h


# 测试：用序列数据验证GRU前向传播
if __name__ == "__main__":
    # 配置参数
    input_dim = 3    # 输入特征维度
    hidden_dim = 2   # 隐藏状态维度
    seq_len = 4      # 序列长度（时间步数）
    
    # 初始化GRU
    gru = GRU(input_dim, hidden_dim)
    
    # 生成随机输入序列（seq_len个时间步，每个时间步输入维度为input_dim）
    x_seq = np.random.randn(seq_len, 1, input_dim)  # 形状 (seq_len, 1, input_dim)
    
    # 初始化隐藏状态（初始为0）
    h_prev = np.zeros((1, hidden_dim))
    
    # 按时间步执行GRU前向传播
    print("时间步 | 隐藏状态h")
    print("-" * 30)
    for t in range(seq_len):
        x_t = x_seq[t]  # 当前时间步输入
        h_prev = gru.forward(x_t, h_prev)  # 更新隐藏状态
        print(f"  {t}   | {h_prev.round(4)}")
```

