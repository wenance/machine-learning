'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''



from __future__ import print_function
from numpy import genfromtxt
import numpy as np

import argparse

import tensorflow as tf

def normalize(data):
    np.nor

# Import MNIST data
from trainer import input_data

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--bucket',
      help='Bucket to work with ',
      required=True
  )

  parser.add_argument(
      '--job-dir',
      help='Location to put the outputs',
      required=True
  )

  args = parser.parse_args()


print ("Staging bucket is ", args.bucket)
#mnist = input_data.read_data_sets(args.bucket + "data" , one_hot=True)

NUM_TRAIN = 8000
NUM_FEATURES = 9
BATCH_SIZE = 100
numpy_data = genfromtxt("../data/reno5.csv", delimiter=',',skip_header=True,dtype='float32')

data_x = numpy_data[0:NUM_TRAIN,0:NUM_FEATURES]
data_y = np.reshape(numpy_data[0:NUM_TRAIN,NUM_FEATURES],(-1,1))

data_x_test = numpy_data[NUM_TRAIN:,0:NUM_FEATURES]
data_y_test = np.reshape(numpy_data[NUM_TRAIN:,NUM_FEATURES],(-1,1))


#Normalizar...
data_x = data_x / np.linalg.norm(data_x,axis=0)
data_x_test = data_x_test / np.linalg.norm(data_x_test,axis=0)

total_batch = np.size(data_y,axis=0) / BATCH_SIZE

# Parameters
learning_rate = 0.01
training_epochs = 1300
batch_size = 100
display_step = 1

x = tf.placeholder(tf.float32, [None, NUM_FEATURES]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 1]) # solo el valor de si esta en mora o no

dx = tf.data.Dataset.from_tensor_slices(data_x)
dy = tf.data.Dataset.from_tensor_slices(data_y)
dcomb_train = tf.data.Dataset.zip((dx, dy)).batch(BATCH_SIZE)
train_iterator = dcomb_train.make_initializable_iterator()
next_train = train_iterator.get_next()



# Set model weights
W1 = tf.Variable(tf.zeros([NUM_FEATURES,10]))
b1 = tf.Variable(tf.zeros([10]))

W2 = tf.Variable(tf.zeros([10, 1]))
b2 = tf.Variable(tf.zeros([1]))

# Construct model
pred = tf.nn.tanh(tf.matmul(x, W1) + b1) # TANH, una sola salida
pred = tf.nn.sigmoid(tf.matmul(pred, W2) + b2) # TANH, una sola salida


cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=pred)
cost = tf.reduce_mean(cross_entropy)
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))

# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(np.size(data_x,axis=0)/batch_size)
        # Loop over all batches
        sess.run(train_iterator.initializer)

        for i in range(total_batch):
            (batch_xs, batch_ys) = sess.run(next_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        # Test model
        correct_prediction = tf.equal(tf.round (pred), y)
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({x: data_x_test, y: data_y_test}))
        comparison = np.array( [pred.eval({x: data_x_test, y: data_y_test}), data_y_test])
        print ()

    print("Optimization Finished!")