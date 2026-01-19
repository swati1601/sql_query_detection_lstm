import numpy as np
import pandas as pd


data = pd.read_csv("sql_injection_dataset_large.csv")

queries = data["query"].values
labels = data["label"].values


def text_to_sequence(text, max_len=30):
    seq = [ord(c) for c in text[:max_len]]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq

X = np.array([text_to_sequence(q) for q in queries])
y = np.array(labels)



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

input_size = 1
hidden_size = 16
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


# LSTM Cell

def lstm_cell(x, h_prev, c_prev):
    x = np.array([[x]])

    f = sigmoid(Wf @ x + bf)
    i = sigmoid(Wi @ x + bi)
    o = sigmoid(Wo @ x + bo)
    c_hat = tanh(Wc @ x + bc)

    c = f * c_prev + i * c_hat
    h = o * tanh(c)

    return h, c

# Training



for epoch in range(100):
    total_loss = 0

    for x_seq, y_true in zip(X, y):

        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        for value in x_seq:
            h, c = lstm_cell(value, h, c)

        y_pred = sigmoid(Wy @ h + by)

        # loss
        if y_true == 1:
            loss = 5 * (y_pred - y_true) ** 2
        else:
            loss = (y_pred - y_true) ** 2

        total_loss += loss

        #output layer
        error = y_pred - y_true
        Wy -= learning_rate * error * h.T
        by -= learning_rate * error

    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss[0][0]}")


# Prediction

def predict(query):
    seq = text_to_sequence(query)
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    for v in seq:
        h, c = lstm_cell(v, h, c)

    out = sigmoid(Wy @ h + by)
    return 1 if out > 0.5 else 0

#testing
#print( predict("admin' --"))  
print( predict("SELECT email FROM customers WHERE customer_id = 105"))
#print(predict("' OR EXISTS(SELECT * FROM users) --"))
print(predict('AND 1=1 --'))
