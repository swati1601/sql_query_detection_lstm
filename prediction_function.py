def predict(query):
    seq = text_to_sequence(query)
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))

    for v in seq:
        h, c = lstm_cell(v, h, c)

    out = sigmoid(Wy @ h + by)
    return 1 if out > 0.5 else 0

