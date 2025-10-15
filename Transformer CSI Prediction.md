# Transformer CSI Prediction

## 代码实现

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import math
import warnings

warnings.filterwarnings('ignore')


class CSIDataset(Dataset):
    """CSI数据集类"""

    def __init__(self, data, window_length=10, prediction_length=1):
        self.data = data
        self.window_length = window_length
        self.prediction_length = prediction_length
        self.samples = self._create_samples()

    def _create_samples(self):
        """创建训练样本"""
        samples = []
        for i in range(len(self.data) - self.window_length - self.prediction_length + 1):
            # 历史窗口
            input_seq = self.data[i:i + self.window_length]
            # 未来序列
            target_seq = self.data[i + self.window_length:i + self.window_length + self.prediction_length]
            samples.append((input_seq, target_seq))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_seq, target_seq = self.samples[idx]
        return torch.FloatTensor(input_seq), torch.FloatTensor(target_seq)


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_seq_length=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        seq_length = x.size(1)
        return x + self.pe[:seq_length, :].unsqueeze(0)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """缩放点积注意力"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换并分割为多个头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 应用注意力
        attention_output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # 连接多个头
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)

        # 应用输出线性变换
        output = self.W_o(attention_output)

        return output


class TransformerBlock(nn.Module):
    """Transformer块"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 多头注意力 + 残差连接 + 层归一化
        attention_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    """Transformer解码器"""

    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoder, self).__init__()

        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return x


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.cross_attention = MultiHeadAttention(d_model, num_heads)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # 掩码自注意力
        self_attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # cross 编码器-解码器注意力
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        # x:自解码器自注意力后的输出（当前解码状态）, encoder_output:来自编码器的输出（输入序列的特征）
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x


class TransformerChannelPredictor(nn.Module):
    """基于Transformer的信道预测模型"""

    def __init__(self, input_dim=2, d_model=128, num_heads=2, num_layers=2,
                 d_ff=256, prediction_length=1, dropout=0.1):
        super(TransformerChannelPredictor, self).__init__()

        self.d_model = d_model
        self.prediction_length = prediction_length

        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)

        # 位置编码
        self.positional_encoding = PositionalEncoding(d_model)

        # 编码器
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, d_ff, dropout)

        # 解码器
        self.decoder = TransformerDecoder(num_layers, d_model, num_heads, d_ff, dropout)

        # 输出投影层
        self.output_projection = nn.Linear(d_model, input_dim)

        # 解码器输入初始化
        self.decoder_start = nn.Parameter(torch.randn(1, 1, d_model))

    def create_masks(self, src, tgt):
        """创建注意力掩码"""
        src_mask = None  # 编码器不需要掩码

        # 解码器掩码（防止看到未来信息）
        tgt_len = tgt.size(1)
        tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len)).unsqueeze(0)

        return src_mask, tgt_mask.to(tgt.device)

    def forward(self, src, prediction_mode='multi_step', guided_training=False, target=None):
        """
        Args:
            src: 输入序列 (batch_size, seq_length, input_dim)
            prediction_mode: 'multi_step' 或 'recursive'
            guided_training: 是否使用引导训练
            target: 目标序列（仅在引导训练时使用）
        """
        batch_size = src.size(0)

        # 编码器
        src_embedded = self.input_projection(src)
        src_embedded = self.positional_encoding(src_embedded)
        encoder_output = self.encoder(src_embedded)

        if prediction_mode == 'multi_step':
            return self._multi_step_prediction(encoder_output, batch_size)
        else:
            return self._recursive_prediction(encoder_output, batch_size, guided_training, target)

        def _multi_step_prediction(self, encoder_output, batch_size):
            k = self.prediction_length
            # 用一个“起始token”扩展成长度k的序列（或用最后一个历史时刻的投影）
            start = self.decoder_start.expand(batch_size, 1, -1)        # [B,1,d_model]
            decoder_input = start.repeat(1, k, 1).contiguous()          # [B,k,d_model]
            src_mask, tgt_mask = self.create_masks(encoder_output, decoder_input)
            dec_out = self.decoder(decoder_input, encoder_output, src_mask, tgt_mask)  # 两层decoder级联
            return self.output_projection(dec_out)                      # [B,k,2]


    def _recursive_prediction(self, encoder_output, batch_size, guided_training=False, target=None):
        """递归预测"""
        outputs = []

        # 初始化解码器输入
        decoder_input = self.decoder_start.expand(batch_size, 1, -1)

        for i in range(self.prediction_length):
            # 解码器前向传播
            decoder_output = self.decoder(decoder_input, encoder_output)

            # 输出投影
            output = self.output_projection(decoder_output)
            outputs.append(output)

            # 准备下一时刻的输入
            if guided_training and target is not None and i < self.prediction_length - 1:
                # 引导训练：使用真实值
                next_input = self.input_projection(target[:, i:i + 1, :])
            else:
                # 非引导训练：使用预测值
                next_input = self.input_projection(output)
            decoder_input = torch.cat([decoder_seq, next_input], dim=1)     # 关键：把新步拼到序列尾部

        return torch.cat(outputs, dim=1)


def generate_synthetic_csi_data(num_samples=5000, seq_length=50):
    """生成合成CSI数据用于演示"""
    t = np.linspace(0, 100, num_samples)

    # 模拟复杂的信道变化
    real_part = np.sin(0.1 * t) + 0.3 * np.sin(0.3 * t) + 0.1 * np.random.randn(num_samples)
    imag_part = np.cos(0.15 * t) + 0.2 * np.cos(0.25 * t) + 0.1 * np.random.randn(num_samples)

    # 组合实部和虚部
    csi_data = np.stack([real_part, imag_part], axis=1)

    return csi_data


def calculate_evm(predictions, targets):
    """计算误差向量幅度(EVM)"""
    # 重构复数
    pred_complex = predictions[:, :, 0] + 1j * predictions[:, :, 1]
    target_complex = targets[:, :, 0] + 1j * targets[:, :, 1]

    # 计算EVM
    error_power = np.sum(np.abs(pred_complex - target_complex) ** 2)
    signal_power = np.sum(np.abs(target_complex) ** 2)

    evm = np.sqrt(error_power / signal_power) * 100
    return evm


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3):
    """训练模型"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # 前向传播（使用多步预测）
            output = model(data, prediction_mode='multi_step')
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, prediction_mode='multi_step')
                val_loss += criterion(output, target).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return train_losses, val_losses


def evaluate_model(model, test_loader):
    """评估模型性能"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_predictions = []
    all_targets = []
    all_rmse = []
    all_r2 = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            # 多步预测
            output = model(data, prediction_mode='multi_step')

            # 转换为numpy数组
            pred_np = output.cpu().numpy()
            target_np = target.cpu().numpy()

            all_predictions.append(pred_np)
            all_targets.append(target_np)

            # 计算每个批次的指标
            for i in range(pred_np.shape[0]):
                # RMSE (分别计算实部和虚部)
                rmse_real = np.sqrt(mean_squared_error(target_np[i, :, 0], pred_np[i, :, 0]))
                rmse_imag = np.sqrt(mean_squared_error(target_np[i, :, 1], pred_np[i, :, 1]))
                rmse_avg = (rmse_real + rmse_imag) / 2
                all_rmse.append(rmse_avg)

                # R2-Score
                r2_real = r2_score(target_np[i, :, 0], pred_np[i, :, 0])
                r2_imag = r2_score(target_np[i, :, 1], pred_np[i, :, 1])
                r2_avg = (r2_real + r2_imag) / 2
                all_r2.append(r2_avg)

    # 合并所有预测和目标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # 计算整体指标
    avg_rmse = np.mean(all_rmse)
    avg_r2 = np.mean(all_r2)
    evm = calculate_evm(all_predictions, all_targets)

    return {
        'RMSE': avg_rmse,
        'R2_Score': avg_r2,
        'EVM': evm,
        'predictions': all_predictions,
        'targets': all_targets
    }


def visualize_results(results, num_samples=100):
    """可视化预测结果"""
    predictions = results['predictions'][:num_samples]
    targets = results['targets'][:num_samples]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 实部预测结果
    axes[0, 0].plot(targets[:, 0, 0], label='True Real Part', alpha=0.7)
    axes[0, 0].plot(predictions[:, 0, 0], label='Predicted Real Part', alpha=0.7)
    axes[0, 0].set_title('Real Part Prediction')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 虚部预测结果
    axes[0, 1].plot(targets[:, 0, 1], label='True Imaginary Part', alpha=0.7)
    axes[0, 1].plot(predictions[:, 0, 1], label='Predicted Imaginary Part', alpha=0.7)
    axes[0, 1].set_title('Imaginary Part Prediction')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 误差分析
    error_real = np.abs(predictions[:, 0, 0] - targets[:, 0, 0])
    error_imag = np.abs(predictions[:, 0, 1] - targets[:, 0, 1])

    axes[1, 0].plot(error_real, label='Real Part Error')
    axes[1, 0].plot(error_imag, label='Imaginary Part Error')
    axes[1, 0].set_title('Prediction Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 误差分布
    axes[1, 1].hist(error_real, alpha=0.5, label='Real Part Error', bins=30)
    axes[1, 1].hist(error_imag, alpha=0.5, label='Imaginary Part Error', bins=30)
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("基于Transformer的信道预测模型训练")
    print("=" * 50)

    # 设置参数
    window_length = 10
    prediction_length = 5
    batch_size = 32
    num_epochs = 100
    learning_rate = 1e-3

    # 生成合成数据
    print("1. 生成合成CSI数据...")
    csi_data = generate_synthetic_csi_data(num_samples=5000)

    # 创建数据集
    dataset = CSIDataset(csi_data, window_length, prediction_length)

    # 划分数据集
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size])

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"数据集大小: 训练集 {len(train_dataset)}, 验证集 {len(val_dataset)}, 测试集 {len(test_dataset)}")

    # 创建模型
    print("\n2. 创建Transformer模型...")
    model = TransformerChannelPredictor(
        input_dim=2,
        d_model=128,
        num_heads=2,
        num_layers=2,
        d_ff=256,
        prediction_length=prediction_length,
        dropout=0.1
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # 训练模型
    print("\n3. 开始训练...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs, learning_rate)

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 评估模型
    print("\n4. 评估模型性能...")
    results = evaluate_model(model, test_loader)

    print(f"\n测试集性能指标:")
    print(f"RMSE: {results['RMSE']:.6f}")
    print(f"R²-Score: {results['R2_Score']:.4f}")
    print(f"EVM: {results['EVM']:.2f}%")

    # 可视化结果
    print("\n5. 可视化预测结果...")
    visualize_results(results)

    return model, results


# 比较不同模型的函数
class SimpleRNN(nn.Module):
    """简单RNN模型用于比较"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, prediction_length=1):
        super(SimpleRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # RNN前向传播
        out, _ = self.rnn(x, h0)

        # 使用最后一个时刻的输出进行预测
        last_output = out[:, -1, :]  # (batch_size, hidden_dim)

        # 生成多步预测
        predictions = []
        current_hidden = last_output
        for _ in range(self.prediction_length):
            pred = self.fc(current_hidden)
            predictions.append(pred.unsqueeze(1))
            # 递归时，current_hidden = last_output 或加 hidden 投影
            # current_hidden = last_output
        return torch.cat(predictions, dim=1)


class SimpleLSTM(nn.Module):
    """简单LSTM模型用于比较"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=2, prediction_length=1):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.prediction_length = prediction_length

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(x.device)

        # LSTM前向传播
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # 使用最后一个时刻的输出进行预测
        last_output = out[:, -1, :]  # (batch_size, hidden_dim)

        # 生成多步预测
        predictions = []
        current_hidden = last_output
        for _ in range(self.prediction_length):
            pred = self.fc(current_hidden)
            predictions.append(pred.unsqueeze(1))
            # 递归时，current_hidden 仍然用 last_output 或者 pred 投影回 hidden_dim
            # 这里推荐用 current_hidden = last_output
            # 或者如果想用 pred 递归，需要加一层 hidden 投影
            # current_hidden = some_hidden_layer(pred)
            # 但最简单的做法是 current_hidden = last_output
        return torch.cat(predictions, dim=1)


class SimpleDNN(nn.Module):
    """简单DNN模型用于比较"""

    def __init__(self, input_dim=2, window_length=10, hidden_dim=128, prediction_length=1):
        super(SimpleDNN, self).__init__()
        self.prediction_length = prediction_length

        self.fc1 = nn.Linear(input_dim * window_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim * prediction_length)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)

        # 展平输入
        x = x.view(batch_size, -1)

        # DNN前向传播
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # 重塑输出
        x = x.view(batch_size, self.prediction_length, -1)

        return x


def compare_models():
    """比较不同模型的性能"""
    print("\n模型性能比较")
    print("=" * 50)

    # 参数设置
    window_length = 10
    prediction_length = 5
    batch_size = 32
    num_epochs = 50  # 减少训练轮数以节省时间

    # 生成数据
    csi_data = generate_synthetic_csi_data(num_samples=3000)
    dataset = CSIDataset(csi_data, window_length, prediction_length)

    # 划分数据集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建不同模型
    models = {
        'Transformer': TransformerChannelPredictor(
            input_dim=2, d_model=128, num_heads=2, num_layers=2,
            d_ff=256, prediction_length=prediction_length
        ),
        'LSTM': SimpleLSTM(input_dim=2, hidden_dim=128, num_layers=2, prediction_length=prediction_length),
        'RNN': SimpleRNN(input_dim=2, hidden_dim=128, num_layers=2, prediction_length=prediction_length),
        'DNN': SimpleDNN(input_dim=2, window_length=window_length, hidden_dim=128, prediction_length=prediction_length)
    }

    results = {}

    for name, model in models.items():
        print(f"\n训练 {name} 模型...")

        # 训练模型
        if name == 'Transformer':
            # Transformer使用特殊的训练函数
            train_losses, _ = train_model(model, train_loader, test_loader, num_epochs // 2, 1e-3)
        else:
            # 其他模型使用简单训练
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for epoch in range(num_epochs // 2):
                model.train()
                train_loss = 0
                for data, target in train_loader:
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()

                if epoch % 10 == 0:
                    print(f'Epoch {epoch}, Loss: {train_loss / len(train_loader):.6f}')

        # 评估模型
        if name == 'Transformer':
            model_results = evaluate_model(model, test_loader)
        else:
            # 其他模型的简单评估
            model.eval()
            all_predictions = []
            all_targets = []

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)

                    all_predictions.append(output.cpu().numpy())
                    all_targets.append(target.cpu().numpy())

            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)

            # 计算指标
            rmse_real = np.sqrt(mean_squared_error(all_targets[:, :, 0].flatten(), all_predictions[:, :, 0].flatten()))
            rmse_imag = np.sqrt(mean_squared_error(all_targets[:, :, 1].flatten(), all_predictions[:, :, 1].flatten()))
            rmse_avg = (rmse_real + rmse_imag) / 2

            r2_real = r2_score(all_targets[:, :, 0].flatten(), all_predictions[:, :, 0].flatten())
            r2_imag = r2_score(all_targets[:, :, 1].flatten(), all_predictions[:, :, 1].flatten())
            r2_avg = (r2_real + r2_imag) / 2

            evm = calculate_evm(all_predictions, all_targets)

            model_results = {
                'RMSE': rmse_avg,
                'R2_Score': r2_avg,
                'EVM': evm
            }

        results[name] = model_results
        print(
            f"{name} - RMSE: {model_results['RMSE']:.6f}, R²: {model_results['R2_Score']:.4f}, EVM: {model_results['EVM']:.2f}%")

    # 绘制比较结果
    plt.figure(figsize=(15, 5))

    models_names = list(results.keys())
    rmse_values = [results[name]['RMSE'] for name in models_names]
    r2_values = [results[name]['R2_Score'] for name in models_names]
    evm_values = [results[name]['EVM'] for name in models_names]

    plt.subplot(1, 3, 1)
    plt.bar(models_names, rmse_values)
    plt.title('RMSE Comparison')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 2)
    plt.bar(models_names, r2_values)
    plt.title('R² Score Comparison')
    plt.ylabel('R² Score')
    plt.xticks(rotation=45)

    plt.subplot(1, 3, 3)
    plt.bar(models_names, evm_values)
    plt.title('EVM Comparison')
    plt.ylabel('EVM (%)')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    return results


def hyperparameter_analysis():
    """超参数分析"""
    print("\n超参数分析")
    print("=" * 50)

    # 生成数据
    csi_data = generate_synthetic_csi_data(num_samples=2000)

    # 分析时间窗口长度的影响
    window_lengths = [5, 10, 15, 20, 25]
    window_results = []

    for window_length in window_lengths:
        print(f"测试窗口长度: {window_length}")

        dataset = CSIDataset(csi_data, window_length, 1)  # 预测长度固定为1
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        # 创建并训练模型
        model = TransformerChannelPredictor(
            input_dim=2, d_model=64, num_heads=2, num_layers=1,
            d_ff=128, prediction_length=1
        )

        # 简化训练过程
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # 快速训练
        for epoch in range(20):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data, prediction_mode='multi_step')
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

        # 评估
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data, prediction_mode='multi_step')
                test_loss += criterion(output, target).item()

        avg_test_loss = test_loss / len(test_loader)
        window_results.append(avg_test_loss)
        print(f"测试损失: {avg_test_loss:.6f}")

    # 绘制窗口长度分析结果
    plt.figure(figsize=(10, 6))
    plt.plot(window_lengths, window_results, 'o-')
    plt.xlabel('Window Length')
    plt.ylabel('Test Loss (MSE)')
    plt.title('Impact of Window Length on Model Performance')
    plt.grid(True)
    plt.show()

    return window_lengths, window_results


if __name__ == "__main__":
    # 运行主程序
    model, results = main()

    # 比较不同模型
    comparison_results = compare_models()

    # 超参数分析
    window_analysis = hyperparameter_analysis()

    print("\n实验完成！")
    print("这个实现展示了基于Transformer的信道预测的完整流程，包括：")
    print("1. 数据预处理和特征工程")
    print("2. Transformer网络架构实现")
    print("3. 训练策略（引导vs非引导训练）")
    print("4. 性能评估和可视化")
    print("5. 与其他深度学习模型的比较")
    print("6. 超参数分析")
