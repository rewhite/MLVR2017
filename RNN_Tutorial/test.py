# -*- coding: utf-8 -*-
"""
rnn1.py
Created on Fri May 12 11:07:06 2017

@author: Cho
"""
import tensorflow as tf
import numpy as np

char_rdic = ['h', 'e', 'l', 'o'] # id -> char
char_dic = {w : i for i, w in enumerate(char_rdic)} # char -> id
print (char_dic)

ground_truth = [char_dic[c] for c in 'hello']
print (ground_truth)

x_data = np.array([[1,0,0,0], # h
                   [0,1,0,0], # e
                   [0,0,1,0], # l
                   [0,0,1,0]], # l
                 dtype = 'f')

x_data = tf.one_hot(ground_truth[:-1], len(char_dic), 1.0, 0.0, -1)
print(x_data) #4x4

# Configuration
rnn_size = len(char_dic) # 4
batch_size = 1
output_size = 4

# RNN Model
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units = rnn_size,
                                       #input_size = None, # deprecated at tensorflow 0.9
                                       #activation = tanh,
                                       )

initial_state = rnn_cell.zero_state(batch_size, tf.float32)
initial_state_1 = tf.zeros([batch_size, rnn_cell.state_size]) #  위 코드와 같은 결과

x_split = tf.split(x_data, len(char_dic), 0) # 가로축으로 4개로 split

print(x_split)

sess = tf.Session()
print(sess.run(x_data))
print(sess.run(x_split))


outputs, state = tf.contrib.rnn.static_rnn(rnn_cell, x_split, initial_state)

print (outputs)
# print (state)

logits = tf.reshape(tf.concat(outputs, 1), # shape = 1 x 16
                    [-1, rnn_size])        # shape = 4 x 4
logits.get_shape()

targets = tf.reshape(ground_truth[1:], [-1]) # a shape of [-1] flattens into 1-D
targets.get_shape()

weights = tf.ones([len(char_dic) * batch_size])

loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char_rdic[t] for t in result])
