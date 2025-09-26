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
