import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv("sql_injection_dataset_large.csv")

print(data.head())
data["len"] = data["query"].apply(len)
print(data["len"].describe())


all_text = "".join(data["query"].values)

chars = sorted(list(set(all_text)))
char_to_idx = {ch: i+1 for i, ch in enumerate(chars)}

vocab_size = len(char_to_idx) + 1
print("Vocabulary size:", vocab_size)




MAX_LEN = 40  # fixed length

def encode_query(query):
    encoded = [char_to_idx.get(ch, 0) for ch in query]
    if len(encoded) < MAX_LEN:
        encoded += [0] * (MAX_LEN - len(encoded))
    else:
        encoded = encoded[:MAX_LEN]
    return encoded



class SQLDataset(Dataset):
    def __init__(self, dataframe):
        self.queries = dataframe["query"].values
        self.labels = dataframe["label"].values

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        x = encode_query(self.queries[idx])
        x = torch.tensor(x, dtype=torch.long)   
        y = torch.tensor(self.labels[idx], dtype=torch.float)
        return x, y


dataset = SQLDataset(data)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
x, y = next(iter(loader))
print("x shape:", x.shape)
print("x dtype:", x.dtype)
print("y shape:", y.shape)




class SQLLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=128,
            
            batch_first=True
        )
        #self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        #out = self.dropout(out)
        return self.fc(out)
model = SQLLSTM()
print(model)



num_normal = len(data[data.label == 0])
num_inject = len(data[data.label == 1])

pos_weight = torch.tensor([num_normal / num_inject])

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=0.003)



print("Before:", model.fc.weight.data[0][:5])

print("After:", model.fc.weight.data[0][:5])

model.train()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCEWithLogitsLoss()

for epoch in range(15):
    total_loss = 0
    for x, y in loader:
        optimizer.zero_grad()

        y_pred = model(x).view(-1)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader)}")


THRESHOLD = 0.5

def predict(query):
    encoded = encode_query(query)
    tensor = torch.tensor([encoded], dtype=torch.long)

    model.eval()
    with torch.no_grad():
        raw = model(tensor)          
        prob = torch.sigmoid(raw)    

    prob_value = prob.item()

    if prob_value >= THRESHOLD:
        label = 1
        result = "SQL Injection"
    else:
        label = 0
        result = "Normal Query"

    print("QUERY :", query)
   # print("PROB  :", prob_value)
    print("LABEL :", label)
    print("RESULT:", result)


predict("SELECT * FROM users WHERE id=1")
predict("admin' OR '1'='1")
predict("update employees set salary=5000 where id=3")
predict("drop database testdb")
predict("delete from logs where created_at < '2026-01-01'")

predict("select * from inventory where stock > 0")

predict("update employees set salary=5000 where id=3")

predict("insert into feedback(user_id, message) values(1,'good')")
predict("select credit_card from users")

predict("update accounts set balance=0 where id=5")



predict("insert into users(username,password) values('admin','123')")

predict("' UNION SELECT username, password FROM users --")

predict("union select username, password from login")
predict("delete from orders where order_id=10")
