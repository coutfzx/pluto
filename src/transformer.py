import torch
import torch.nn as nn
import math

"""
Transformer架构详解与实现

Transformer是2017年Google提出的序列到序列模型，完全基于注意力机制，摒弃了传统的RNN/CNN结构。
它在自然语言处理任务中取得了巨大成功，如BERT、GPT等模型都基于Transformer架构。

架构组成：
1. 编码器（Encoder）：由N个相同的层堆叠而成
2. 解码器（Decoder）：由N个相同的层堆叠而成
3. 每个编码器层包含：多头自注意力机制 + 前馈神经网络
4. 每个解码器层包含：多头自注意力 + 多头编码器-解码器注意力 + 前馈神经网络

核心组件：
- 多头自注意力（Multi-Head Self-Attention）
- 位置编码（Positional Encoding）
- 残差连接（Residual Connection）
- 层归一化（Layer Normalization）
- 前馈神经网络（Feed-Forward Network）
"""

class PositionalEncoding(nn.Module):
    """
    位置编码：由于Transformer没有循环或卷积结构，无法捕捉序列的位置信息，
    因此需要添加位置编码来引入序列中单词的相对或绝对位置信息。
    
    位置编码使用正弦和余弦函数的线性组合：
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    其中pos是位置，i是维度
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
      1. torch.arange(0, max_len): 创建从0到max_len-1的一维张量，表示序列中每个位置的索引
      2. .unsqueeze(1): 在第1维扩展一个维度，将形状从[max_len]变成[max_len, 1]，为后续的广播运算做准备
      3. .float(): 将整数张量转换为浮点张量，因为位置编码计算涉及浮点运算

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                            -(math.log(10000.0) / d_model))
      1.  torch.arange(0, d_model, 2) - 创建偶数索引 [0, 2, 4, ...]，对应公式中的 2i
      2. -(math.log(10000.0) / d_model) - 负对数形式的缩放因子
      3. torch.exp(...) - 计算指数，得到最终的分母项

        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        
        # 注册为buffer，不作为模型参数进行优化
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]  # 添加位置编码


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制：允许模型同时关注来自不同表示子空间的信息。
    
    步骤：
    1. 将输入线性变换为Q, K, V
    2. 将Q, K, V分割成多个头
    3. 对每个头计算注意力
    4. 将所有头的输出拼接并线性变换
    """
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # 线性变换权重
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 线性变换并分割成多头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用掩码（用于解码器中的未来掩码或填充掩码）
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 计算输出
        output = torch.matmul(attention_weights, V)
        
        # 重新拼接多头
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # 最终线性变换
        return self.W_o(output)


class FeedForward(nn.Module):
    """
    前馈神经网络：对每个位置独立地应用相同的前馈网络。
    通常包含两个线性变换和一个激活函数。
    """
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    """
    编码器的一个层，包含多头自注意力和前馈网络
    每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderLayer(nn.Module):
    """
    解码器的一个层，包含三个子层：
    1. 自注意力（掩码）
    2. 编码器-解码器注意力
    3. 前馈网络
    每个子层都有残差连接和层归一化
    """
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力（掩码） + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 编码器-解码器注意力 + 残差连接 + 层归一化
        enc_dec_output = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(enc_dec_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x


class Encoder(nn.Module):
    """
    编码器：由N个相同的编码器层堆叠而成
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过所有编码器层
        for layer in self.layers:
            x = layer(x, mask)
            
        return x


class Decoder(nn.Module):
    """
    解码器：由N个相同的解码器层堆叠而成
    """
    def __init__(self, vocab_size, d_model, num_layers, num_heads, d_ff, max_len=5000, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过所有解码器层
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
            
        return x


class Transformer(nn.Module):
    """
    完整的Transformer模型：编码器 + 解码器 + 输出层
    """
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_layers=6, num_heads=8, d_ff=2048, max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_layers, num_heads, d_ff, max_len, dropout)
        
        # 输出层：将解码器输出映射到目标词汇表大小
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器输出
        enc_output = self.encoder(src, src_mask)
        
        # 解码器输出
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        
        # 映射到词汇表
        output = self.output_proj(dec_output)
        return self.softmax(output)
    
    def generate_square_subsequent_mask(self, sz):
        """
        生成用于解码器自注意力的后续掩码，防止模型看到未来的信息
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


# 示例：创建一个小型Transformer模型
def create_transformer_model():
    """
    创建一个示例Transformer模型实例
    """
    # 模型参数
    src_vocab_size = 1000  # 源语言词汇表大小
    tgt_vocab_size = 1000  # 目标语言词汇表大小
    d_model = 512          # 模型维度
    num_layers = 6         # 编码器/解码器层数
    num_heads = 8          # 注意力头数
    d_ff = 2048            # 前馈网络隐藏层维度
    max_len = 5000         # 最大序列长度
    dropout = 0.1          # dropout概率
    
    # 创建模型
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout
    )
    
    return model


# 模型参数统计
def count_parameters(model):
    """
    统计模型参数数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# 模型使用示例
if __name__ == "__main__":
    print("=== Transformer架构详解与实现 ===\n")
    
    # 创建模型
    model = create_transformer_model()
    print(f"模型创建成功！")
    print(f"总参数量: {count_parameters(model):,}")
    
    # 模型结构说明
    print("\n=== 模型架构说明 ===")
    print("1. 编码器 (Encoder):")
    print("   - 输入: 源序列 (src)")
    print("   - 处理: N个编码器层，每层包含多头自注意力 + 前馈网络")
    print("   - 输出: 编码后的源序列表示")
    
    print("\n2. 解码器 (Decoder):")
    print("   - 输入: 目标序列 (tgt) + 编码器输出")
    print("   - 处理: N个解码器层，每层包含自注意力 + 编码器-解码器注意力 + 前馈网络")
    print("   - 输出: 解码后的目标序列表示")
    
    print("\n3. 关键组件:")
    print("   - 位置编码: 引入序列位置信息")
    print("   - 多头注意力: 捕捉不同子空间的依赖关系")
    print("   - 残差连接: 缓解梯度消失问题")
    print("   - 层归一化: 稳定训练过程")
    print("   - 前馈网络: 非线性变换")
    
    # 模拟输入
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 8
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    
    src = torch.randint(0, src_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, tgt_vocab_size, (batch_size, tgt_seq_len))
    
    # 生成掩码
    src_mask = None  # 实际应用中可能需要掩码填充部分
    tgt_mask = model.generate_square_subsequent_mask(tgt_seq_len).unsqueeze(0).repeat(batch_size, 1, 1)
    
    # 前向传播
    output = model(src, tgt, src_mask, tgt_mask)
    
    print(f"\n=== 前向传播测试 ===")
    print(f"输入形状 - 源序列: {src.shape}, 目标序列: {tgt.shape}")
    print(f"输出形状: {output.shape} (批次, 序列长度, 词汇表大小)")
    print(f"模型成功运行，输出维度正确！")
    
    # 模型各部分参数量
    print(f"\n=== 各部分参数量 ===")
    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    output_proj_params = count_parameters(model.output_proj)
    
    print(f"编码器参数量: {encoder_params:,}")
    print(f"解码器参数量: {decoder_params:,}")
    print(f"输出投影层参数量: {output_proj_params:,}")
    print(f"(注意: 各部分参数量可能因实现细节略有差异)")



