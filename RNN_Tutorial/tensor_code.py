import tensorflow as tf
import numpy as np
# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

print(chars)
# hyperparameters
hidden_size = vocab_size #100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
Wxh = np.random.randn(hidden_size, vocab_size)*0.01 # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*0.01 # hidden to output
bh = np.zeros((hidden_size, 1)) # hidden bias
by = np.zeros((vocab_size, 1)) # output bias

# Wxh = tf.get_variable("Wxh", shape=[hidden_size, vocab_size],
#            initializer=tf.contrib.layers.xavier_initializer())
# test = "Hello"

# preprocessing
n,p = 0,0
inputs = []
targets = []

while p+seq_length+1 <= len(data):
    inputs.append([char_to_ix[ch] for ch in data[p:p+seq_length]])
    targets.append([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]])
    p += seq_length

sess = tf.Session()

one_hot_data = tf.one_hot(inputs,vocab_size)
target_one_hot = tf.one_hot(targets, vocab_size)
# print(sess.run(one_hot_data))
# print(sess.run(one_hot_data[0]))

# print(sess.run(tf.split(one_hot_data[0],25,0)))
splited_data = tf.split(one_hot_data[0],seq_length,0)

# print(splited_data)
rnn_cell = tf.contrib.rnn.BasicRNNCell(hidden_size) #100


# print(splited_data)


#backup
# state = tf.zeros((1,hidden_size)) #100x
# state = rnn_cell.zero_state(1,dtype=tf.float32)
# output, state = tf.nn.static_rnn(rnn_cell,splited_data,state)
# # print("asdas")
# sess.run(tf.global_variables_initializer())
# print(sess.run(output))
# print(targets)
# print(target_one_hot)
# loss = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=target_one_hot[0])
# #
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
#
# sess.run(tf.global_variables_initializer())
# for i in range(500):
#     sess.run(optimizer)
#     print(sess.run(tf.argmax(output,1)))



# print(state)

# # outputs = []
# for input_ in inputs:
# #     # print(input_)
# #     # output, state = tf.nn.static_rnn(rnn_cell,input_)
#     input_ = tf.constant(['3','3'])
#     output, state = rnn_cell(input_, state)
# #     # outputs.append(output)

# #preprocessing
# for a in test:
#     print(char_to_ix[a])


# a = [0,1,2,3]
# print(char_to_ix)








# rnn_cell = tf.contrib.rnn.BasicRNNCell(vocab_size) #25

# state = tf.zeros((hidden_size,rnn_cell.state_size))
#
# char_rdic = ['h', 'e', 'l', 'o'] # id -> char
# char_dic = {w : i for i, w in enumerate(char_rdic)} # char -> id
# print (char_dic)
#
# x_data = np.array([[1,0,0,0], # h
#                    [0,1,0,0], # e
#                    [0,0,1,0], # l
#                    [0,0,1,0]], # l
#                  dtype = 'f')
# ground_truth = [char_dic[c] for c in 'hello']
# x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)
#
#
# print(x_data)
# x_split = tf.split(x_data, len(char_dic), 0) # 가로축으로 4개로 split
# print(x_split)
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(rnn_cell.state_size))

# initial_state = state = rnn_cell.zero_state(hidden_size, tf.float32)

# preprocessing
# n,p = 0,0
# inputs = []
# targets = []
# while p+seq_length+1 <= len(data):
#     inputs.append([char_to_ix[ch] for ch in data[p:p+seq_length]])
#     targets.append([char_to_ix[ch] for ch in data[p+1:p+seq_length]])
#     p += seq_length
#
# # outputs = []
# for input_ in inputs:
# #     # print(input_)
# #     # output, state = tf.nn.static_rnn(rnn_cell,input_)
#     input_ = tf.constant(['3','3'])
#     output, state = rnn_cell(input_, state)
# #     # outputs.append(output)

# x = tf.constant([[1]], dtype = tf.float32)
# x2 = tf.constant([[0]], dtype = tf.float32)
# rnn_cell = tf.contrib.rnn.BasicRNNCell(2)
# lstm_cell = tf.contrib.rnn.BasicLSTMCell(2)
#
# outputs1, states1 = rnn_cell(rnn_cell, [x,x2,x2,x2,x2,x2,x2,x2,x2])
# outputs2, states2 = rnn_cell(lstm_cell, [x,x2,x2,x2,x2,x2,x2,x2,x2])
# init = tf.global_variables_initializer()
# print(inputs)
# output, state = tf.contrib.rnn.static_rnn(rnn_cell,inputs,initial_state)


# stacked_rnn = tf.contrib.rnn.rnn_cell.MultiRNNCell(
#    [rnn_cell() for _ in range(seq_length)])
#
# initial_state = state = stacked_rnn.zero_state(data_size, tf.float32)


# input_x = tf.placeholder()
# cost =
# optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost)

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# print(sess.run(initial_state))
# sess.close()
