# Transformer CSI Prediction

## 代码实现

### 1.数据预处理 (data_loader.py)

```python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class CSIDataset(Dataset):
    def __init__(self, data, L, k):
        self.L = L  # 历史窗口长度
        self.k = k  # 预测窗口长度

        # 假设data是 (N, 2) 的NumPy数组, N是总时间点, 2是实部和虚部
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data) - self.L - self.k + 1

    def __getitem__(self, idx):
        # 输入序列 X
        src = self.data[idx : idx + self.L]
        # 目标序列 Y
        tgt = self.data[idx + self.L : idx + self.L + self.k]
        return src, tgt

# --- 使用示例 ---
# 1. 加载你的CSI数据 (例如从.csv文件)
# csi_data = load_csi_data('beijing_tianjin_hsr.csv')
# csi_data_real_imag = np.vstack((csi_data.real, csi_data.imag)).T

# 2. 归一化 (非常重要)
# scaler = MinMaxScaler()
# csi_data_scaled = scaler.fit_transform(csi_data_real_imag)

# 3. 创建数据集和加载器
# L, k = 10, 5 # 历史10步，预测5步
# train_dataset = CSIDataset(train_data_scaled, L, k)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
```

### 2.模型定义 (model.py)

```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerCSIModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, output_dim, k):
        super(TransformerCSIModel, self).__init__()
        self.d_model = d_model
        self.k = k # 预测长度

        # 输入层 (将2维的CSI映射到d_model维)
        self.encoder_input_layer = nn.Linear(input_dim, d_model)
        self.decoder_input_layer = nn.Linear(input_dim, d_model)

        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # Transformer 主体
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True # PyTorch 1.9+ 支持
        )

        # 输出层 (将d_model维的输出映射回2维的CSI)
        self.output_layer = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt):
        # src: (batch_size, L, input_dim)
        # tgt: (batch_size, k, input_dim)

        # 1. 嵌入和位置编码
        src_embedded = self.pos_encoder(self.encoder_input_layer(src))
        tgt_embedded = self.pos_encoder(self.decoder_input_layer(tgt))

        # 2. 创建解码器的掩码 (防止看到未来信息)
        tgt_mask = self.transformer.generate_square_subsequent_mask(self.k).to(src.device)

        # 3. 通过Transformer
        # 注意：PyTorch的Transformer默认需要 (seq_len, batch_size, d_model)
        # 如果使用 batch_first=True, 则输入为 (batch_size, seq_len, d_model)
        # 这里假设使用 batch_first=True
        output = self.transformer(src_embedded, tgt_embedded, tgt_mask=tgt_mask)

        # 4. 通过输出层得到预测结果
        predictions = self.output_layer(output)

        return predictions
```

### 3. 训练脚本 (train.py)

```python
# --- 模型参数 (根据论文结论设置) ---
INPUT_DIM = 2
D_MODEL = 128
N_HEAD = 2
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 512 # 通常是d_model的2-4倍
OUTPUT_DIM = 2
K = 5 # 假设预测未来5步

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransformerCSIModel(INPUT_DIM, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_DIM, K).to(device)

criterion = nn.MSELoss() # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# --- 训练循环 ---
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for src, tgt_real in train_loader:
        src, tgt_real = src.to(device), tgt_real.to(device)

        # 在多步预测中，解码器的输入通常是目标序列的"shifted right"版本
        # 这里简单地用目标序列本身作为输入，因为模型内部有掩码
        tgt_input = tgt_real 

        optimizer.zero_grad()

        # 前向传播
        predictions = model(src, tgt_input)

        # 计算损失
        loss = criterion(predictions, tgt_real)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

# --- 保存模型 ---
# torch.save(model.state_dict(), 'transformer_csi_model.pth')
```

## 完整实现论文 'Transformer Network Based Channel Prediction for CSI Feedback Enhancement in AI-Native Air Interface'

```

"""
该脚本包含:
1.  一个合成CSI数据集，模拟无线信道的时间相关性。
2.  一个标准的PyTorch Dataset类，用于处理时序数据。
3.  四个模型的定义：DNN, RNN, LSTM, 和 Transformer。 (新增)
4.  完整的训练和评估循环。
5.  三种可视化功能：
    a. 单样本预测对比图 (复现图10的部分思想)
    b. 长序列预测对比图 (新增，复现图10)
    c. 多模型性能对比图 (新增，复现图9)
"""

import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import time

# ==========================================================================================
# 1. 配置参数 (Configuration)
# ==========================================================================================
class Config:
    L = 10 
    K = 5
    INPUT_DIM = 2
    OUTPUT_DIM = 2
    D_MODEL = 128
    N_HEAD = 2
    NUM_ENCODER_LAYERS = 2
    NUM_DECODER_LAYERS = 2
    DIM_FEEDFORWARD = 512
    # --- 新增：为RNN/LSTM/DNN设置隐藏层维度 ---
    HIDDEN_DIM_RNN = 128 

    BATCH_SIZE = 64
    NUM_EPOCHS = 30 # 为了快速演示，减少epoch，实际可增加
    LEARNING_RATE = 0.0001
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"当前使用的设备 (Device): {Config.DEVICE}")


# ==========================================================================================
# 2. 数据生成 (Data Generation)
# ==========================================================================================
def generate_synthetic_csi_data(num_samples=5000):
    print("正在生成合成CSI数据...")
    time_steps = np.arange(0, num_samples)
    real_part = (np.sin(0.05 * time_steps) * 1.0 + np.sin(0.20 * time_steps) * 0.5 + np.sin(0.50 * time_steps) * 0.25) + np.random.normal(0, 0.1, num_samples)
    imag_part = (np.cos(0.05 * time_steps + np.pi/4) * 1.0 + np.cos(0.22 * time_steps) * 0.45 + np.sin(0.48 * time_steps) * 0.3) + np.random.normal(0, 0.1, num_samples)
    csi_data = np.stack([real_part, imag_part], axis=1)
    print(f"数据生成完毕，形状为: {csi_data.shape}")
    return csi_data


# ==========================================================================================
# 3. PyTorch Dataset 类 (Dataset Class)
# ==========================================================================================
class CSIDataset(Dataset):
    def __init__(self, data, L, K):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.L = L
        self.K = K

    def __len__(self):
        return len(self.data) - self.L - self.K + 1

    def __getitem__(self, idx):
        src = self.data[idx : idx + self.L]
        tgt = self.data[idx + self.L : idx + self.L + self.K]
        return src, tgt


# ==========================================================================================
# 4. 模型定义 (Model Definition) - 新增DNN, RNN, LSTM
# ==========================================================================================

# --- Transformer 模型 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerCSIModel(nn.Module):
    def __init__(self, config):
        super(TransformerCSIModel, self).__init__()
        self.config = config
        self.encoder_input_layer = nn.Linear(config.INPUT_DIM, config.D_MODEL)
        self.decoder_input_layer = nn.Linear(config.INPUT_DIM, config.D_MODEL)
        self.pos_encoder = PositionalEncoding(config.D_MODEL)
        self.transformer = nn.Transformer(d_model=config.D_MODEL, nhead=config.N_HEAD, num_encoder_layers=config.NUM_ENCODER_LAYERS, num_decoder_layers=config.NUM_DECODER_LAYERS, dim_feedforward=config.DIM_FEEDFORWARD, batch_first=True)
        self.output_layer = nn.Linear(config.D_MODEL, config.OUTPUT_DIM)

    def forward(self, src, tgt):
        src_embedded = self.encoder_input_layer(src)
        tgt_embedded = self.decoder_input_layer(tgt)
        src_pos = self.pos_encoder(src_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_pos = self.pos_encoder(tgt_embedded.permute(1, 0, 2)).permute(1, 0, 2)
        tgt_mask = self.transformer.generate_square_subsequent_mask(self.config.K).to(self.config.DEVICE)
        output = self.transformer(src_pos, tgt_pos, tgt_mask=tgt_mask)
        predictions = self.output_layer(output)
        return predictions

# --- LSTM 模型 ---
class LSTMModel(nn.Module):
    def __init__(self, config):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(config.INPUT_DIM, config.HIDDEN_DIM_RNN, batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM_RNN, config.K * config.OUTPUT_DIM)
        self.config = config

    def forward(self, src, tgt=None): # tgt是为了接口统一
        # LSTM只关心历史序列src
        _, (h_n, _) = self.lstm(src)
        # 取最后一层的隐藏状态并送入全连接层
        output = self.fc(h_n[-1])
        # 将输出reshape成(batch, K, output_dim)以匹配目标
        return output.view(-1, self.config.K, self.config.OUTPUT_DIM)

# --- RNN 模型 ---
class RNNModel(nn.Module):
    def __init__(self, config):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(config.INPUT_DIM, config.HIDDEN_DIM_RNN, batch_first=True)
        self.fc = nn.Linear(config.HIDDEN_DIM_RNN, config.K * config.OUTPUT_DIM)
        self.config = config

    def forward(self, src, tgt=None):
        _, h_n = self.rnn(src)
        output = self.fc(h_n[-1])
        return output.view(-1, self.config.K, self.config.OUTPUT_DIM)

# --- DNN 模型 ---
class DNNModel(nn.Module):
    def __init__(self, config):
        super(DNNModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc_layers = nn.Sequential(
            nn.Linear(config.L * config.INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.K * config.OUTPUT_DIM)
        )
        self.config = config

    def forward(self, src, tgt=None):
        # DNN将历史序列展平，然后通过全连接层
        src_flat = self.flatten(src)
        output = self.fc_layers(src_flat)
        return output.view(-1, self.config.K, self.config.OUTPUT_DIM)

# ==========================================================================================
# 5. 训练和评估函数
# ==========================================================================================
def train_model(model, train_loader, criterion, optimizer, config):
    model.train()
    total_loss = 0
    for src, tgt_real in train_loader:
        src, tgt_real = src.to(config.DEVICE), tgt_real.to(config.DEVICE)
        tgt_input = tgt_real
        optimizer.zero_grad()
        predictions = model(src, tgt_input)
        loss = criterion(predictions, tgt_real)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# --- 新增：用于图9的评估函数，计算每个预测步的RMSE ---
def evaluate_for_rmse_plot(model, test_loader, config):
    model.eval()
    # 存储每个预测步长的总平方误差
    total_squared_errors = np.zeros(config.K)
    num_samples = 0

    with torch.no_grad():
        for src, tgt_real in test_loader:
            src, tgt_real = src.to(config.DEVICE), tgt_real.to(config.DEVICE)
            tgt_input = tgt_real

            predictions = model(src, tgt_input)

            # 计算每个样本在每个预测步长的平方误差
            squared_errors = (predictions - tgt_real).pow(2).cpu().numpy()
            # 在实部和虚部维度上求和，得到每个预测步的误差
            squared_errors = np.sum(squared_errors, axis=2)
            # 累加到总误差中
            total_squared_errors += np.sum(squared_errors, axis=0)
            num_samples += src.size(0)

    # 计算均方根误差 (RMSE)
    mean_squared_errors = total_squared_errors / num_samples
    rmse_per_step = np.sqrt(mean_squared_errors)
    return rmse_per_step


# ==========================================================================================
# 6. 可视化函数 - 新增两个绘图函数
# ==========================================================================================
def plot_single_sample_prediction(model, data_loader, scaler, config):
    # 此函数与原代码中的plot_results功能相同
    model.eval()
    src, tgt_real = next(iter(data_loader))
    src, tgt_real = src.to(config.DEVICE), tgt_real.to(config.DEVICE)

    with torch.no_grad():
        predictions_scaled = model(src, tgt_real).cpu().numpy()

    tgt_real_scaled = tgt_real.cpu().numpy()

    sample_idx = 0
    predictions_original = scaler.inverse_transform(predictions_scaled[sample_idx])
    tgt_real_original = scaler.inverse_transform(tgt_real_scaled[sample_idx])

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 1, 1)
    plt.plot(range(config.K), tgt_real_original[:, 0], 'b-o', label='真实值 (Real Part)')
    plt.plot(range(config.K), predictions_original[:, 0], 'r--x', label='预测值 (Real Part)')
    plt.title(f'单一样本预测结果对比 (模型: {model.__class__.__name__})')
    plt.ylabel('幅值')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(range(config.K), tgt_real_original[:, 1], 'b-o', label='真实值 (Imaginary Part)')
    plt.plot(range(config.K), predictions_original[:, 1], 'r--x', label='预测值 (Imaginary Part)')
    plt.xlabel(f'未来 {config.K} 个时间步')
    plt.ylabel('幅值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 新增：用于复现图10的函数 ---
def plot_long_sequence_prediction(model, full_dataset, scaler, config, num_points_to_plot=1200):
    """
    在一段连续的长序列上进行预测并可视化，以复现图10的效果。
    """
    model.eval()

    # 从完整数据集中提取一段连续的数据
    true_values_scaled = full_dataset.data[:num_points_to_plot].numpy()

    predictions_scaled = np.zeros((num_points_to_plot, config.OUTPUT_DIM))

    with torch.no_grad():
        for i in range(num_points_to_plot - config.L - config.K):
            # 构造输入序列
            src = full_dataset.data[i : i + config.L].unsqueeze(0).to(config.DEVICE)
            # 为了预测，解码器也需要一个输入，这里我们简单地用一个零矩阵
            # 更好的做法是进行真正的递归预测，但为了快速可视化，这里简化处理
            tgt_input = torch.zeros(1, config.K, config.INPUT_DIM).to(config.DEVICE)

            prediction_step = model(src, tgt_input).cpu().numpy()

            # 我们只记录第一个预测步的结果，以形成连续的预测曲线
            predictions_scaled[i + config.L] = prediction_step[0, 0, :]

    # 填充预测序列的前L个点，以便绘图
    predictions_scaled[:config.L] = true_values_scaled[:config.L]

    # 反归一化
    true_values_original = scaler.inverse_transform(true_values_scaled)
    predictions_original = scaler.inverse_transform(predictions_scaled)

    plt.figure(figsize=(15, 8))

    # 绘制实部
    plt.subplot(2, 1, 1)
    plt.plot(true_values_original[:, 0], 'b-', label='真实值 (Real Part)', alpha=0.7)
    plt.plot(predictions_original[:, 0], 'r-', label='预测值 (Real Part)', alpha=0.8)
    plt.title(f'长序列预测对比 (复现图10) - 模型: {model.__class__.__name__}')
    plt.ylabel('幅值')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, num_points_to_plot)

    # 绘制虚部
    plt.subplot(2, 1, 2)
    plt.plot(true_values_original[:, 1], 'b-', label='真实值 (Imaginary Part)', alpha=0.7)
    plt.plot(predictions_original[:, 1], 'r-', label='预测值 (Imaginary Part)', alpha=0.8)
    plt.xlabel('时间步 (Snapshot Index)')
    plt.ylabel('幅值')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, num_points_to_plot)

    plt.tight_layout()
    plt.show()

# --- 新增：用于复现图9的函数 ---
def plot_model_comparison_rmse(results, config):
    """
    绘制多模型在不同预测长度下的RMSE对比图，复现图9。
    """
    plt.figure(figsize=(10, 6))

    markers = ['-o', '-s', '-^', '-D']
    for i, (model_name, rmse_values) in enumerate(results.items()):
        plt.plot(range(1, config.K + 1), rmse_values, markers[i], label=model_name)

    plt.title('多模型性能对比 (复现图9)')
    plt.xlabel('预测长度 (Prediction Length)')
    plt.ylabel('均方根误差 (RMSE)')
    plt.xticks(range(1, config.K + 1))
    plt.legend()
    plt.grid(True)
    plt.show()


# ==========================================================================================
# 7. 主执行流程
# ==========================================================================================
if __name__ == '__main__':
    config = Config()

    csi_data = generate_synthetic_csi_data()
    scaler = MinMaxScaler(feature_range=(-1, 1))
    csi_data_scaled = scaler.fit_transform(csi_data)

    train_size = int(len(csi_data_scaled) * 0.8)
    train_data = csi_data_scaled[:train_size]
    test_data = csi_data_scaled[train_size:]

    train_dataset = CSIDataset(train_data, config.L, config.K)
    test_dataset = CSIDataset(test_data, config.L, config.K)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # --- 训练和评估所有模型 ---
    models = {
        'Transformer': TransformerCSIModel(config),
        'LSTM': LSTMModel(config),
        'RNN': RNNModel(config),
        'DNN': DNNModel(config)
    }

    model_rmse_results = {}

    for name, model in models.items():
        print(f"\n--- 正在处理模型: {name} ---")
        model.to(config.DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

        start_time = time.time()
        print("开始训练...")
        for epoch in range(config.NUM_EPOCHS):
            train_model(model, train_loader, criterion, optimizer, config)
        end_time = time.time()
        print(f"训练完成，耗时: {end_time - start_time:.2f} 秒")

        print("正在评估...")
        rmse_per_step = evaluate_for_rmse_plot(model, test_loader, config)
        model_rmse_results[name] = rmse_per_step
        print(f"{name} 在各预测步长的RMSE: {rmse_per_step}")

        # 只为Transformer模型绘制详细的预测图
        if name == 'Transformer':
            print("为Transformer模型生成详细预测图...")
            full_csi_dataset = CSIDataset(csi_data_scaled, config.L, config.K)
            plot_long_sequence_prediction(model, full_csi_dataset, scaler, config)

    # --- 绘制最终的对比图 ---
    print("\n绘制多模型性能对比图 (复现图9)...")
    plot_model_comparison_rmse(model_rmse_results, config)
```


