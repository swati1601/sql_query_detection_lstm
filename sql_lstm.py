import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# load dataset
data = pd.read_csv("sql_injection_dataset_large.csv")


# character vocabulary
all_text = "".join(data["query"].values)
chars = sorted(set(all_text))
char_to_idx = {c: i + 1 for i, c in enumerate(chars)}

vocab_size = len(char_to_idx) + 1
max_len = 40


def encode_query(query):
    seq = [char_to_idx.get(c, 0) for c in query]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq[:max_len]


class SQLDataset(Dataset):
    def __init__(self, df):
        self.queries = df["query"].values
        self.labels = df["label"].values

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        x = torch.tensor(encode_query(self.queries[idx]), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


dataset = SQLDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)


class SQLLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 50)
        self.lstm = nn.LSTM(50, 100, batch_first=True)
        self.fc = nn.Linear(100, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


model = SQLLSTM()

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# training

model.train()

for epoch in range(15):
    total_loss = 0

    for x, y in loader:
        optimizer.zero_grad()
        output = model(x).squeeze()
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print("Epoch", epoch + 1, "Loss:", total_loss / len(loader))


#prediction


def predict(query):
    model.eval()
    x = torch.tensor([encode_query(query)], dtype=torch.long)

    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()

    if prob >= 0.5:
        print(query, "-> SQL Injection")
    else:
        print(query, "-> Normal Query")


# testing  queries
predict("SELECT * FROM users WHERE id=1")
predict("admin' OR '1'='1")
predict("' UNION SELECT username, password FROM users --")
predict("insert into users values('admin','123')")
predict("delete from orders where id=5")

