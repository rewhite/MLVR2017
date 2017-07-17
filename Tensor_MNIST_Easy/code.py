import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# mnist is a lightweight class which stores the training, validation,
# and testing sets as NumPy arrays.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.initialize_all_variables())

y = tf.nn.softmax(tf.matmul(x,W) + b)

#tf.reduce_sum은 모든 클래스에 대해 결과를 합하는 함수, tf.reduce_mean은 사용된 이미지들 각각에서 계산된 합의 평균을 구하는 함수
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#학습 속도 0.5의 경사 하강법(steepest gradient descent) 알고리즘을 사용하여 크로스 엔트로피를 최소화
#TensorFlow가 실제로 하는 것은 계산 그래프에 기울기를 계산하고, 얼마나 매개변수를 변경해야 할지 계산하고,
#매개변수를 변경하는 새로운 계산들을 추가하는 것
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#반환된 train_step은 실행되었을 때 경사 하강법을 통해 각각의 매개변수를 변화시키게 됩니다.
#따라서, 모델을 훈련시키려면 이 train_step을 반복해서 실행하면 됩니다.
for i in range(1000):
  batch = mnist.train.next_batch(50)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
