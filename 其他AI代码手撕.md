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
