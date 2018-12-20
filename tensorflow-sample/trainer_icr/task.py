'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import argparse
import numpy as np
import tensorflow as tf

# Import MNIST data

from tensorflow.examples.tutorials.mnist import input_data

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

print("Staging bucket is ", args.bucket)
mnist = input_data.read_data_sets(args.bucket + "data", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10])  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
epoch_costs = np.empty(0)
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
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch

        epoch_costs = np.append(epoch_costs, avg_cost)
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

    import matplotlib.pyplot as plt
    import numpy as np

    # Data for plotting
    t = np.arange(0, np.size(epoch_costs))

    fig, ax = plt.subplots()
    ax.plot(t, epoch_costs)

    ax.set(xlabel='Epoch number', ylabel='Cross entropy cost function values',
           title='Evolution of the Cost function over the epoch iterations')
    # ax.grid()

    #fig.savefig("test.png")
    plt.show()

    yy = pred.eval({x: mnist.test.images})

    for i in range(5):
        digit = np.argmax(yy[i], axis=0)
        title = "El digito detectado es:" + str(digit) + " con probabilidad: " + '{0:.2f} %'.format(yy[i, digit] * 100)
        print(title)
        plt.title(title)
        plt.imshow(np.array(mnist.test.images[i]).reshape((28, 28)) * 255)
        plt.show()
