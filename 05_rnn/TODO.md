# 05_rnn TODO List

## 项目结构重构

从 RNN 开始采用模块化结构：

```
05_rnn/
├── configs/
│   └── rnn_config.yaml          # 超参数配置
├── layers/
│   ├── __init__.py
│   ├── rnn.py                   # RNNCell, LSTM, GRU
│   ├── dense.py                 # Dense (可从04_cnn复用)
│   ├── embedding.py             # 词嵌入层
│   └── activations.py           # tanh, sigmoid, softmax
├── models/
│   ├── __init__.py
│   ├── rnn_lm.py                # RNN 语言模型
│   └── lstm_classifier.py       # LSTM 分类器
├── utils/
│   ├── __init__.py
│   ├── data.py                  # 数据加载、序列处理
│   ├── logger.py                # 日志系统
│   ├── metrics.py               # 困惑度、准确率
│   └── trainer.py               # 通用训练器
├── train.py                     # 训练入口 (CLI)
├── evaluate.py                  # 评估脚本
└── README.md
```

---

## 训练优化 (从 CNN 延续)

### 1. 向量化数据增强
- [ ] `random_crop` 去掉 for 循环，用批量索引

```python
# 当前 (慢)
for i in range(batch):
    crops.append(X_padded[i:i+1, :, top:top+h, left:left+w])

# 优化后 (快)
tops = torch.randint(0, 2*padding+1, (batch,))
lefts = torch.randint(0, 2*padding+1, (batch,))
# 使用 gather 或 advanced indexing
```

### 2. 学习率调度
- [ ] Cosine Annealing (比 step decay 更平滑)

```python
def cosine_lr(epoch, total_epochs, lr_init, lr_min=1e-6):
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + math.cos(epoch / total_epochs * math.pi))
```

- [ ] Warmup (前几个 epoch 线性增加 LR)

```python
def warmup_lr(epoch, warmup_epochs, lr_init):
    if epoch < warmup_epochs:
        return lr_init * (epoch + 1) / warmup_epochs
    return lr_init
```

### 3. 早停 (Early Stopping)
- [ ] 连续 N 个 epoch 无提升则停止

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def __call__(self, val_score):
        if self.best_score is None or val_score > self.best_score + self.min_delta:
            self.best_score = val_score
            self.counter = 0
            return False  # 继续训练
        self.counter += 1
        return self.counter >= self.patience  # True = 停止
```

### 4. 梯度裁剪
- [ ] 防止 RNN 梯度爆炸 (对 RNN 特别重要!)

```python
def clip_grad_norm(parameters, max_norm=1.0):
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm
```

### 5. Label Smoothing
- [ ] 软化标签，防止过拟合

```python
def label_smoothing(y_true, num_classes, epsilon=0.1):
    """
    y_true: (batch,) 整数标签
    return: (batch, num_classes) 软标签
    """
    one_hot = torch.zeros(len(y_true), num_classes, device=y_true.device)
    one_hot.scatter_(1, y_true.unsqueeze(1), 1)
    return one_hot * (1 - epsilon) + epsilon / num_classes
```

---

## RNN 核心实现

### 基础 RNN
- [ ] RNNCell: `h_t = tanh(W_ih @ x_t + W_hh @ h_{t-1} + b)`
- [ ] 时序展开 (BPTT)
- [ ] 双向 RNN (可选)

### LSTM
- [ ] 遗忘门: `f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)`
- [ ] 输入门: `i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)`
- [ ] 输出门: `o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)`
- [ ] 细胞状态: `c_t = f_t * c_{t-1} + i_t * tanh(W_c @ [h_{t-1}, x_t] + b_c)`
- [ ] 隐藏状态: `h_t = o_t * tanh(c_t)`

### GRU (可选)
- [ ] 更新门 + 重置门

---

## 数据集选项

| 任务 | 数据集 | 说明 |
|------|--------|------|
| 字符级语言模型 | Shakespeare / 诗词 | 预测下一个字符 |
| 序列分类 | IMDB 情感分析 | 正面/负面分类 |
| 命名实体识别 | CoNLL-2003 | 序列标注 |

---

## 配置文件示例

```yaml
# configs/rnn_config.yaml
model:
  type: lstm
  hidden_dim: 256
  num_layers: 2
  dropout: 0.3
  bidirectional: false

training:
  batch_size: 64
  epochs: 50
  lr: 0.001
  weight_decay: 1e-5
  grad_clip: 1.0

scheduler:
  type: cosine  # step, cosine, warmup_cosine
  warmup_epochs: 5

early_stopping:
  patience: 10
  min_delta: 0.001

data:
  seq_length: 100
  vocab_size: 10000
```

---

## 通用训练器

```python
# utils/trainer.py
class Trainer:
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.early_stopping = EarlyStopping(config.patience)
        self.best_params = None

    def train(self, train_data, val_data):
        for epoch in range(self.config.epochs):
            lr = self.get_lr(epoch)
            train_loss = self.train_epoch(train_data, lr)
            val_loss, val_acc = self.evaluate(val_data)

            self.logger.log(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_checkpoint()

            if self.early_stopping(val_acc):
                self.logger.log("Early stopping triggered")
                break
```

---

## 优先级

1. **高优先级**
   - [ ] 梯度裁剪 (RNN 必须)
   - [ ] LSTM 实现
   - [ ] 项目结构重构

2. **中优先级**
   - [ ] 通用训练器
   - [ ] Cosine LR + Warmup
   - [ ] Early Stopping

3. **低优先级**
   - [ ] Label Smoothing
   - [ ] 向量化数据增强
   - [ ] GRU 实现
