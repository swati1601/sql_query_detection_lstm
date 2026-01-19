epochs = 50

for epoch in range(epochs):
    loss_sum = 0

    for i in range(len(X)):
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))

        for value in X[i]:
            h, c = lstm_cell(value, h, c)

        y_pred = sigmoid(Wy @ h + by)
        loss = (y_pred - y[i]) ** 2
        loss_sum += loss

        Wy -= learning_rate * (y_pred - y[i]) * h.T
        by -= learning_rate * (y_pred - y[i])

    print(f"Epoch {epoch+1}, Loss: {loss_sum[0][0]}")

