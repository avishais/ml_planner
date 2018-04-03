""" Auto Encoder Example.

Build a 2 layers auto-encoder with TensorFlow to compress images to a
lower latent space and then reconstruct them.

References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.

Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import

from ae_functions import *

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.io as sio
import time
import random

def run_net(X, learning_rate, num_hidden_layer, k, activation):

    n_test = 10000
    n = X.shape[0]-n_test

    x_max = np.max(X)
    x_min = np.min(X)

    x_train = normz(X[0:n,:], x_max, x_min)
    x_test = normz(X[n:,:], x_max, x_min)

    # Training Parameters
    num_steps = 6000
    batch_size = 100

    display_step = 1000

    # Network Parameters
    num_encoded = 6
    hidden_layers = [k]*num_hidden_layer
    hidden_layers.append(num_encoded)
    num_input = X.shape[1] 

    # tf Graph input (only pictures)
    X = tf.placeholder("float", [None, num_input])
    Z = tf.placeholder("float", [None, num_encoded])

    # Store layers weight & bias
    weights, biases = wNb(num_input, hidden_layers)

    # Construct model
    encoder_op = encoder(X, weights, biases, activation)
    decoder_op = decoder(encoder_op, weights, biases, activation)
    decoder_direct_op = decoder(Z, weights, biases, activation)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    # cost = tf.reduce_mean(np.absolute(y_true - y_pred))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Start Training
    # Start a new TF session
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)

        # Training
        for i in range(1, num_steps+1):
            # Get the next batch 
            batch_x = next_batch(batch_size, x_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, c))

        # Save the variables to disk.
        saver.save(sess, "./models/hcp.ckpt")

        # Testing
        # Calculate cost for training data
        x_train_pred = sess.run(decoder_op, {X: x_train})
        z_train_pred = sess.run(encoder_op, {X: x_train})
        training_cost = sess.run(cost, feed_dict={X: x_train})
        
        x_test_pred = sess.run(decoder_op, {X: x_test})
        z_test_pred = sess.run(encoder_op, {X: x_test})
        testing_cost = sess.run(cost, feed_dict={X: x_test})
        
        return training_cost, testing_cost

def log2file(i, learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost):
    f = open('./hparam_abb.txt','a+')

    # f.write("-----------------------------\n")
    # f.write("Trial %d\n" % i)
    # f.write("Learning rate: %f\n" % learning_rate)
    # f.write("Number of hidden layers: %d\n" % num_hidden_layer)
    # f.write("Number of neurons in each layer: %d\n" % k)
    # f.write("Activation: %d" % activation)
    # f.write("Training cost: %f\n" % training_cost)
    # f.write("Testing cost: %f\n" % testing_cost)

    f.write("%d %.5f %d %d %d, %f %f\n" % (i, learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost))

    f.close()


def main():
    #X = np.loadtxt('../data/ckc2d_10_samples_noJL.db')
    X = np.loadtxt('../data/abb_samples_noJL2.db')

    lr = [0.0001,	0.0005,	0.001,	0.002,	0.003,	0.004,	0.005,	0.006,	0.007,	0.008,	0.009,	0.01,	0.02,	0.0288888888888889,	0.0377777777777778,	0.0466666666666667,	0.0555555555555556,	0.0644444444444444,	0.0733333333333333,	0.0822222222222222,	0.0911111111111111,	0.1,	0.110000000000000,	0.120000000000000]

    for i in range(100):
        learning_rate = lr[random.randint(0,len(lr)-1)]
        num_hidden_layer = random.randint(1,5)
        k = random.randint(10,40)
        activation = random.randint(1,3)

        print("-----------------------------")
        print("Trial ", i)
        print("Learning rate: ", learning_rate)
        print("Number of hidden layers: ", num_hidden_layer)
        print("Number of neurons in each layer: ", k)
        if activation==1 : 
            str = "sigmoid" 
        elif activation==2 : 
            str = "relu" 
        else: str = "tanh"
        print("Activation function: ", str)

        training_cost, testing_cost = run_net(X, learning_rate, num_hidden_layer, k, activation)
        print("Training cost: ", training_cost)
        print("Testing cost: ", testing_cost)

        log2file(i, learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost)

if __name__ == '__main__':
  main()