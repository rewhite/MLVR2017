import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) } #{'e': 0, 'o': 1, 'l': 2, 'H': 3, '\n': 4}
ix_to_char = { i:ch for i,ch in enumerate(chars) } #{0: 'e', 1: 'o', 2: 'l', 3: 'H', 4: '\n'}

# hyperparameters
hidden_size = 100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

# print(chars)
# print(data_size)
# print(vocab_size)
# print(char_to_ix)
# print(ix_to_char)

p = 0

inputs = [char_to_ix[ch] for ch in data[p:p+4]]
inputs2 = [ix_to_char[ix] for ix in inputs]
print(inputs)
print(inputs2)
