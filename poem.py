import re
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ====================== 9.4.1 数据预处理 ======================
def process_poems(file_path):
    poems = []
    char_set = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # 去除标点和空白
            line = re.sub(r"[，。？！；：、]", "", line).strip()
            if len(line) < 5:
                continue
            char_set.update(list(line))
            poems.append(list(line))
    # 构建词表
    vocab = list(char_set) + ["<UNK>"]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    # 转为索引序列
    sequences = []
    for poem in poems:
        seq = [word2idx.get(word, word2idx["<UNK>"]) for word in poem]
        sequences.append(seq)
    return sequences, word2idx, vocab

sequences, word2idx, vocab = process_poems(r"D:\poems\poems.txt")

# ====================== 9.4.2 自定义Dataset ======================
class PoetryDataset(Dataset):
    def __init__(self, sequences, seq_len):
        self.seq_len = seq_len
        self.data = []
        for seq in sequences:
            for i in range(0, len(seq) - self.seq_len):
                self.data.append((seq[i:i + self.seq_len], seq[i + 1:i + self.seq_len + 1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.LongTensor(self.data[idx][0])
        y = torch.LongTensor(self.data[idx][1])
        return x, y

dataset = PoetryDataset(sequences, 24)

# ====================== 9.4.3 搭建RNN模型 ======================
class PoetryRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_size=256, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, input, hx=None):
        embed = self.embed(input)
        output, hidden = self.rnn(embed, hx)
        output = self.linear(output)
        return output, hidden

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoetryRNN(len(vocab), embedding_dim=256, hidden_size=512, num_layers=2).to(device)

# ====================== 9.4.4 模型训练 ======================
def train(model, dataset, lr, epoch_num, batch_size, device):
    model.train()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        loss_accumulate = 0
        for batch_count, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            output, _ = model(x)
            # CrossEntropy输入要求：(batch, class, seq_len)
            loss_value = loss_func(output.transpose(1,2), y)
            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            loss_accumulate += loss_value.item()

            # 进度条打印
            print(f"\repoch:{epoch:0>2}[{'='*int((batch_count+1)/len(dataloader)*50):<50}]", end="")
        print(f" loss:{loss_accumulate/len(dataloader):.6f}")

train(model=model, dataset=dataset, lr=1e-3, epoch_num=5, batch_size=32, device=device)

# ====================== 9.4.5 古诗生成函数 ======================
def generate_poem(model, word2idx, vocab, start_token, line_num=4, line_length=7):
    model.eval()
    poem = []
    current_line_length = line_length
    start_token_idx = word2idx.get(start_token, word2idx["<UNK>"])
    if start_token_idx != word2idx["<UNK>"]:
        poem.append(vocab[start_token_idx])
        current_line_length -= 1
    input_tensor = torch.LongTensor([[start_token_idx]]).to(device)
    hidden = None

    with torch.no_grad():
        for _ in range(line_num):
            for interpunction in ["，", "。\n"]:
                while current_line_length > 0:
                    output, hidden = model(input_tensor, hidden)
                    prob = torch.softmax(output[0,0], dim=-1)
                    next_token = torch.multinomial(prob, 1)
                    poem.append(vocab[next_token.item()])
                    input_tensor = next_token.unsqueeze(0)
                    current_line_length -= 1
                poem.append(interpunction)
                current_line_length = line_length
    return "".join(poem)

# 调用生成，以"一"开头，生成4句7言绝句
result = generate_poem(model, word2idx, vocab, start_token="一", line_num=4, line_length=7)
print("\n=====生成古诗=====")
print(result)
