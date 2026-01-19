import numpy as np
import random 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# =============================
# 1. Tokenization
# =============================
def tokenize(text):
    return text.lower().split()

# =============================
# 2. Build Vocabulary
# =============================
def build_vocab(texts):
    vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for text in texts:
        for word in tokenize(text):
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab

# =============================
# 3. Text → Sequence
# =============================
def text_to_sequence(text, vocab, max_len=30):
    tokens = tokenize(text)
    seq = [vocab.get(word, vocab["<UNK>"]) for word in tokens]

    if len(seq) < max_len:
        seq += [vocab["<PAD>"]] * (max_len - len(seq))
    else:
        seq = seq[:max_len]

    return seq

# =============================
# 4. Dataset Class
# =============================
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            text_to_sequence(self.texts[idx], self.vocab),
            dtype=torch.long
        )
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y

# =============================
# 5. Embedding + LSTM Model
# =============================
class EmbeddingLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            embed_dim,
            hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        out = out[:, -1, :]
        out = self.fc(out)
        return self.sigmoid(out)

# =============================
# 6. Load LARGE CSV Dataset
# =============================
data = pd.read_csv("sql_injection_dataset_large.csv")

# Safety: missing values handle
data = data.dropna()

texts = data["query"].astype(str).tolist()
labels = data["label"].astype(int).tolist()

print(f"Total samples: {len(texts)}")

# =============================
# 7. Prepare Data
# =============================
vocab = build_vocab(texts)

dataset = TextDataset(texts, labels, vocab)
loader = DataLoader(dataset, batch_size=32, shuffle=True,generator=torch.Generator().manual_seed(42))

# =============================
# 8. Model Setup
# =============================
model = EmbeddingLSTM(
    vocab_size=len(vocab),
    embed_dim=64,
    hidden_size=15
)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# =============================
# 9. Training Loop
# =============================
epochs = 100

for epoch in range(epochs):
    total_loss = 0
    for X, y in loader:
        y = y.unsqueeze(1)

        preds = model(X)
        loss = criterion(preds, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}]  Loss: {total_loss:.4f}")

# =============================
# 10. Prediction Function
# =============================
def predict(text):
    model.eval()
    seq = torch.tensor(
        text_to_sequence(text, vocab),
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():
        prob = model(seq)
    
    # Threshold 0.5 → 1 = SQL, 0 = Normal
    return 1 if prob.item() > 0.5 else 0


# =============================
# 11. Test Predictions
# =============================

print( predict("select password from admin where id=1"))

print(predict("' OR EXISTS(SELECT * FROM users) --"))
print(predict("update employees set salary=5000 where id=3"))
print(predict("union select username, password from login"))
print(predict("delete from logs where created_at < '2026-01-01'"))
print(predict("drop database testdb"))
