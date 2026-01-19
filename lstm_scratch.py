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

