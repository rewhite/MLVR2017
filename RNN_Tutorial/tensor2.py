import tensorflow as tf
import numpy as np

# data I/O
data = open('input.txt', 'r').read() # should be simple plain text file
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print ('data has %d characters, %d unique.' % (data_size, vocab_size))
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

# hyperparameters
hidden_size = vocab_size #100 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# preprocessing
n,p = 0,0
inputs = []
targets = []

while p+seq_length+1 <= len(data):
    inputs.append([char_to_ix[ch] for ch in data[p:p+seq_length]])
    targets.append([char_to_ix[ch] for ch in data[p+1:p+seq_length+1]])
    p += seq_length

# x = tf.placeholder(tf.int32, [batch_size, vocab_size], name='input_placeholder')
# y = tf.placeholder(tf.int32, [batch_size, vocab_size], name='labels_placeholder')

# print(inputs)
x_one_hot = tf.one_hot(inputs,vocab_size) # [len(inputs) x seq_length x vocab_size]
target_one_hot = tf.one_hot(targets,vocab_size)
# rnn_inputs = [tf.squeeze(i,squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]
x_split = tf.split(x_one_hot[0], seq_length, 0)
print("sadasd")
print(x_one_hot)
print(x_split)
# sess = tf.Session()
# print(sess.run(x_split))

# print(x_one_hot)
cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size) #원래는 100
init_state = cell.zero_state(1, tf.float32)
#
# in_ = tf.expand_dims(x_one_hot,axis=0)
#
# # in_ = tf.transpose(x_one_hot, [1, 0, 2])
#

rnn_outputs, final_state = tf.nn.static_rnn(cell, x_split, init_state)
print(tf.reshape(tf.concat(rnn_outputs,1),[-1, vocab_size])) #(1x (125))

reshape_output = tf.reshape(tf.concat(rnn_outputs,1),[-1, vocab_size])
print(reshape_output)
loss = tf.losses.softmax_cross_entropy(target_one_hot[0],reshape_output)
cost = tf.reduce_sum(loss)
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)
# print(targets[0])
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(optimizer)
        result = sess.run(tf.argmax(reshape_output, 1))
        print(result, "".join([ix_to_char[t] for t in result]))
# rnn_outputs, final_state = tf.nn.static_rnn(cell, in_[0][0], init_state)
#
# print()
#
#
#
# sess = tf.Session()
# print(sess.run(x_one_hot))
