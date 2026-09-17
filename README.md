# RNN 古诗生成 (PyTorch)
> 基于单层/多层RNN实现七言绝句古诗生成，字符级语言模型。

## 技术栈
- Python + PyTorch
- nn.Embedding、RNN、Adam优化器、CrossEntropyLoss
- 字符级文本预处理，自定义Dataset

## 文件说明
- `poetry_generate.py`：完整代码（数据预处理、Dataset、RNN模型、训练、古诗生成）


### 关键参数
- seq_len=24 序列长度
- embedding_dim=256
- hidden_size=512，2层RNN
- lr=1e‑3，batch_size=32


