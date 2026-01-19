import numpy as np
import pandas as pd


data = pd.read_csv("sql_injection_dataset_large.csv")

queries = data["query"].values
labels = data["label"].values

# text to seq  text ko numerical format kr deta hai 
# ord har character ko uske ASCII mai convert krta hai
# max len 30 rakhi q ki query length 30 tak check krega agar query length 30 se kam hogi to max length tak wo padding krega(zero se 30 jagah bharega)
#agar query length max len jyada hui to wo use ignore marega chod dega
# return numerical form mai query ki seq dega
def text_to_sequence(text, max_len=100):
    seq = [ord(c) for c in text[:max_len]]
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))
    return seq

X = np.array([text_to_sequence(q) for q in queries])
#mutiple querys ko  ko matrix form mai represents krega 
y = np.array(labels)
# labels ko represent krega


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# activation function 
#sigmoid function output ko 0 aur 1 ke beech mai deta hai
# sigmoid formula= f(x)=e^x/e^x+1

def tanh(x):
    return np.tanh(x)
# ye bhi activation funt hai ye value ko -1 to +1 ke beech ke range mai deta hai
#tanh = e^X-e^-x/e^x+e^-x


input_size = 1# har time step mai 1 feature
hidden_size = 20# lstm ke 20 units matlab 20 feature learn krega
#hidden size=lstm ki memory capacity
# control the model s learning capacity
learning_rate = 0.05
# model kitna fast seekhe
#weights ko har step mai kitna update krna hai
Wf = np.random.randn(hidden_size, input_size)
# forgate ke weights
Wi = np.random.randn(hidden_size, input_size)
# input gate ke weights
Wo = np.random.randn(hidden_size, input_size)
# output gate ke weights
Wc = np.random.randn(hidden_size, input_size)
#cell ke weights


bf = np.zeros((hidden_size, 1))#forgate bais
bi = np.zeros((hidden_size, 1))#input bais
bo = np.zeros((hidden_size, 1))#output bais
bc = np.zeros((hidden_size, 1))#cell candidate bais

Wy = np.random.randn(1, hidden_size)
by = np.zeros((1, 1))


# LSTM Cell

def lstm_cell(x, h_prev, c_prev):
    x = np.array([[x]])

    f = sigmoid(np.dot(Wf, x) + bf)
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

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss[0][0]}")

def predict(query):
    seq = text_to_sequence(query)
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    for v in seq:
        h, c = lstm_cell(v, h, c)
# har characterek ek krke lstm cell se pass hota hai aur har step pe hidden state h aur cell state c update hote hai aur aakhri h se final output nikalta hai
  
        out = sigmoid(Wy @ h + by)
        return 1 if out > 0.5 else 0

#testing
#print( predict("admin' --"))  
print( predict("SELECT email FROM customers WHERE customer_id = 105"))
#print(predict("' OR EXISTS(SELECT * FROM users) --"))
print(predict('AND 1=1 --'))
                                                                                                                                                                                                                                                                                                                                                
