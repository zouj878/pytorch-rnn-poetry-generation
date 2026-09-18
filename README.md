# RNN Chinese Poetry Generation (PyTorch)
> Character-level language model for generating seven-character quatrains with a single/multi-layer RNN.

## Tech Stack
- Python + PyTorch
- `nn.Embedding`, RNN, Adam Optimizer, CrossEntropyLoss
- Character-level text preprocessing, custom PyTorch Dataset

## File Structure
- `poetry_generate.py`: End-to-end implementation including data preprocessing, custom Dataset, RNN model definition, training loop and poetry inference.

## Hyperparameters
| Parameter | Value | Description |
| ---- | ---- | ---- |
| seq_len | 24 | Input sequence length |
| embedding_dim | 256 | Dimension of character embedding |
| hidden_size | 512 | RNN hidden state dimension |
| num_layers | 2 | Number of stacked RNN layers |
| lr | 1e-3 | Learning rate |
| batch_size | 32 | Batch size |

## Project Overview
This project builds a character-level recurrent neural network to generate classical Chinese seven-character quatrains.
The model learns character patterns from raw poetry corpus. During training, it accepts character sequences and predicts the next character in the sequence. After training completes, the model can generate full poems given an initial seed prompt.

### Key Work
1. Construct character vocabulary and implement text preprocessing pipeline
2. Build custom PyTorch Dataset to split corpus into fixed-length sequences
3. Design multi-layer RNN model with embedding layer for sequence modeling
4. Train the network with cross-entropy loss and Adam optimizer
5. Implement sampling logic for inference and poem generation



