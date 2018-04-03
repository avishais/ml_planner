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

import skopt
from skopt import gp_minimize, forest_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

print('Loading training data...')
X = np.loadtxt('../data/abb_samples_noJL2.db')
n_test = 100000
n = X.shape[0]-n_test
x_max = np.max(X)
x_min = np.min(X)

x_train = normz(X[0:n,:], x_max, x_min)
x_test = normz(X[n:,:], x_max, x_min)
print("Size of:")
print("- Training-set:\t\t{}".format(n))
print("- Test-set:\t\t{}".format(n_test))

def run_net(x):
    
    learning_rate, num_hidden_layer, num_nodes, activation = x[0], x[1], x[2], x[3]

    global X

    print("-----------------------------")
    print("Learning rate: ", learning_rate)
    print("Number of hidden layers: ", num_hidden_layer)
    print("Number of neurons in each layer: ", num_nodes)
    if activation==1 : 
        str = "sigmoid" 
    elif activation==2 : 
        str = "relu" 
    else: str = "tanh"
    print("Activation function: ", str)

    # Training Parameters
    num_steps = 10000
    batch_size = 100

    display_step = 100

    # Network Parameters
    num_encoded = 6
    hidden_layers = [num_nodes]*num_hidden_layer
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
    # saver = tf.train.Saver()

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
            # if i % display_step == 0 or i == 1:
            #     print('Step %i: Minibatch Loss: %f' % (i, c))

        # Save the variables to disk.
        # saver.save(sess, "./models/hcp.ckpt")

        # Testing
        # Calculate cost
        training_cost = sess.run(cost, feed_dict={X: x_train})
        testing_cost = sess.run(cost, feed_dict={X: x_test})
        print('Cost - training %f, testing %f' % (training_cost, testing_cost))

        log2file(learning_rate, num_hidden_layer, num_nodes, activation, training_cost, testing_cost)
        
        return testing_cost

def log2file(learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost):
    f = open('./hparam_abb_skopt.txt','a+')

    # f.write("-----------------------------\n")
    # f.write("Trial %d\n" % i)
    # f.write("Learning rate: %f\n" % learning_rate)
    # f.write("Number of hidden layers: %d\n" % num_hidden_layer)
    # f.write("Number of neurons in each layer: %d\n" % k)
    # f.write("Activation: %d" % activation)
    # f.write("Training cost: %f\n" % training_cost)
    # f.write("Testing cost: %f\n" % testing_cost)

    f.write("%.5f %d %d %d, %f %f\n" % (learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost))

    f.close()


dim_learning_rate = Real(low=1e-6, high=1.2e-1, prior='log-uniform',
                        name='learning_rate')
dim_num_hidden_layer = Integer(low=1, high=5, name='num_hidden_layer')
dim_num_nodes = Integer(low=5, high=60, name='num_nodes')
dim_activation = Integer(low=1, high=3, name='activation')

dimensions = [dim_learning_rate,
            dim_num_hidden_layer,
            dim_num_nodes,
            dim_activation]
default_parameters = [0.02889, 3, 24, 1]

# testing_cost = run_net(default_parameters)

search_result = gp_minimize(func=run_net,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                            n_calls=100,
                            x0=default_parameters)

    # log2file(i, learning_rate, num_hidden_layer, k, activation, training_cost, testing_cost)
