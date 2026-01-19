import numpy as np
import pandas as pd

data = pd.read_csv("sql_injection_dataset_large.csv")

queries = data["query"].values
labels = data["label"].values



def text_to_sequence(text, max_len=20):
    seq = [ord(c) for c in text[:max_len]]   # character → number
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))    # padding
    return seq

X = np.array([text_to_sequence(q) for q in queries])
y = np.array(labels)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)


input_size = 1
hidden_size = 8
learning_rate = 0.01

Wf = np.random.randn(hidden_size, input_size)
Wi = np.random.randn(hidden_size, input_size)
Wo = np.random.randn(hidden_size, input_size)
Wc = np.random.randn(hidden_size, input_size)

bf = np.zeros((hidden_size, 1))
bi = np.zeros((hidden_size, 1))
bo = np.zeros((hidden_size, 1))
bc = np.zeros((hidden_size, 1))

Wy = np.random.randn(1, hidden_size)
by = np.zeros((1, 1))




def lstm_cell(x, h_prev, c_prev):
    x = np.array([[x]])

    f = sigmoid(Wf @ x + bf)
    i = sigmoid(Wi @ x + bi)
    o = sigmoid(Wo @ x + bo)
    c_hat = tanh(Wc @ x + bc)

    c = f * c_prev + i * c_hat
    h = o * tanh(c)

    return h, c




epochs = 50

for epoch in range(epochs):
    loss_sum = 0

    for i in range(len(X)):
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        for value in X[i]:
            h, c = lstm_cell(value, h, c)

        y_pred = sigmoid(Wy @ h + by)
        loss += (y_pred - y[i]) ** 2
        loss_sum += loss

        Wy -= learning_rate * (y_pred - y[i]) * h.T
        by -= learning_rate * (y_pred - y[i])

    print(f"Epoch {epoch+1}, Loss: {loss_sum[0][0]}")





def predict(query):
    seq = text_to_sequence(query)
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    for v in seq:
        h, c = lstm_cell(v, h, c)

    out = sigmoid(Wy @ h + by)
    return 1 if out > 0.5 else 0
print(predict("admin' --"))
print(predict("SELECT email FROM customers WHERE customer_id = 105"))
print(predict("' OR EXISTS(SELECT * FROM users) --"))






