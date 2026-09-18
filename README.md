RNN Poetry Generation (PyTorch)
Character-level language model for generating seven-character quatrains using single/multi-layer RNN.

Tech Stack
Python + PyTorch
nn.Embedding, RNN, Adam Optimizer, CrossEntropyLoss
Character-level text preprocessing, custom Dataset

File Description
poetry_generate.py: End-to-end code including data preprocessing, Dataset definition, RNN model, training pipeline and poetry generation.

Hyperparameters
seq_len = 24 (sequence length)
embedding_dim = 256
hidden_size = 512, 2-layer RNN
lr = 1e-3, batch_size = 32


