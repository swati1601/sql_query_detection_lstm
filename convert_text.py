def text_to_sequence(text, max_len=20):
    seq = [ord(c) for c in text[:max_len]]   # character → number
    if len(seq) < max_len:
        seq += [0] * (max_len - len(seq))    # padding
    return seq

X = np.array([text_to_sequence(q) for q in queries])
y = np.array(labels)

