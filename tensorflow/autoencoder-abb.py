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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-retrain", help="Retrain existing model", action="store_true")
parser.add_argument("-plot", help="Plot trained models", action="store_true")
args = parser.parse_args()
if args.retrain and args.plot:
    training = True
    retrain = True
if args.retrain:
    training = True
    retrain = True
elif args.plot:
    training = False
    retrain = False
else:
    training = True
    retrain = False

print('Loading training data...')
if not training and not retrain:
    X = np.loadtxt('../data/abb_samples_noJL_short.db')
else:
    X = np.loadtxt('../data/abb_samples_noJL.db')
# X = X[:int(1e5),:]
n_test = 10000
n = X.shape[0]-n_test

x_max = np.max(X)
x_min = np.min(X)

x_train = normz(X[0:n,:], x_max, x_min)
x_test = normz(X[n:,:], x_max, x_min)
# x_train = X[0:n,:];
# x_test = (X[n:,:])

# plt.figure(0)
# ax = plt.axes(projection='3d')
# npr = int(1e4)
# ir = [0,1,3]#np.random.choice(9, 3)
# print(ir)
# # ax.plot3D(x_train[:npr,0], x_train[:npr,1], x_train[:npr,3], 'ro', markersize=3)
# ax.plot3D(x_test[:npr,ir[0]], x_test[:npr,ir[1]], x_test[:npr,ir[2]], 'bo')
# plt.show()
# exit()

# Training Parameters
learning_rate = 0.09
num_steps = int(1e5)
batch_size = 150

display_step = 1000

# Network Parameters
num_encoded = 6
hidden_layers = [14]*1
hidden_layers.append(num_encoded)
num_input = 12 
activation = 1

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
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

'''
model_abb_4.ckpt - [24]*3, sigmoid, lr=0.0002, passed +200k epochs * => net_abb4.netxt
model_abb_2.ckpt - [42]*2, sigmoid, lr=0.001, passed +200k epochs * => net_abb2.netxt
'''

load_from = 'model_abb_3.ckpt'
save_to = 'model_abb_3.ckpt'

# Start Training
# Start a new TF session
COSTS = []	# for plotting
STEPS = []	# for plotting
start = time.time()
with tf.Session() as sess:

    if training:
    
        if  not retrain:
            # Run the initializer
            sess.run(init)
        else:
            # Restore variables from disk.
            saver.restore(sess, "./models/" + load_from)                
            print("Loaded saved model: %s" % "./models/" + load_from)

        # Training
        for i in range(1, num_steps+1):
            # Get the next batch 
            batch_x = next_batch(batch_size, x_train)

            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x})
            # Display logs per step
            if i % display_step == 0 or i == 1:
                print('Step %i: Minibatch Loss: %f' % (i, c))
                save_path = saver.save(sess, "./models/cp_abb.ckpt")
                COSTS.append(c)
                STEPS.append(i)

        # Save the variables to disk.
        save_path = saver.save(sess, "./models/" + save_to)
        print("Model saved in path: %s" % save_path)

        # Plot cost convergence
        plt.figure(4)
        plt.semilogy(STEPS, COSTS, 'k-')
        plt.xlabel('Step')
        plt.ylabel('Cost')
        plt.ylim([0, np.max(COSTS)])
        plt.grid(True)
    else:
        # Restore variables from disk.
        saver.restore(sess, "./models/" + load_from)

    # Testing
    # Calculate cost for training data
    x_train_pred = sess.run(decoder_op, {X: x_train})
    z_train_pred = sess.run(encoder_op, {X: x_train})
    print("Training cost:", sess.run(cost, feed_dict={X: x_train}))

    x_test_pred = sess.run(decoder_op, {X: x_test})
    z_test_pred = sess.run(encoder_op, {X: x_test})
    testing_cost = sess.run(cost, feed_dict={X: x_test})
    print("Testing cost=", testing_cost)

    # Motion in z-space to C-space
    # z1 = np.linspace(-0.05, -0.81, 50); z1 = np.reshape(z1, (50, 1))
    # z2 = np.linspace(0.8, -0.68, 50); z2 = np.reshape(z2, (50, 1))
    # z_path = np.concatenate((z1,z2), axis=1)
    # x_path = sess.run(decoder_direct_op, {Z: z_path})

    # Grid in z-space to C-space
    # rx = np.linspace(-1, 1, 100); #z1 = np.reshape(z1, (100, 1))
    # ry = np.linspace(-1, 1, 100); #z1 = np.reshape(z1, (100, 1))
    # z_path = []
    # for i in range(len(rx)):
    #     for j in range(len(ry)):
    #         z_path.append((rx[i],ry[j]))
    # z_path = np.array(z_path)
    # x_path = sess.run(decoder_direct_op, {Z: z_path})

    export_net(weights, biases, x_max, x_min, sess)

x_train = denormz(x_train, x_max, x_min)
x_train_pred = denormz(x_train_pred, x_max, x_min)
x_test = denormz(x_test, x_max, x_min)
x_test_pred = denormz(x_test_pred, x_max, x_min)
# x_path = denormz(x_path, x_max, x_min)

# Log path to simulator
# log2path(x_path)

# Log in and out data for analysis
# dataINnOUT(x_train, x_train_pred, z_train_pred)

npr = int(1e4)

# Plots    
fig = plt.figure(1)
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax.plot3D(x_train[:npr,0], x_train[:npr,1], x_train[:npr,2], 'bo')
ax.plot3D(x_train_pred[0:npr,0], x_train_pred[0:npr,1], x_train_pred[0:npr,2], 'go', label='Projection')
# ax.plot3D(x_path[:,0], x_path[:,1], x_path[:,2], 'mo-', label='Projection', markersize=5)
ax.set_title("theta_1, theta_2, theta_3")
ax.grid(True)

ax = fig.add_subplot(2, 2, 2, projection='3d')
ax.plot3D(x_train[:npr,0], x_train[:npr,2], x_train[:npr,3], 'bo')
ax.plot3D(x_train_pred[0:npr,0], x_train_pred[0:npr,2], x_train_pred[0:npr,3], 'go', label='Projection')
# ax.plot3D(x_path[:,0], x_path[:,2], x_path[:,3], 'mo-', label='Projection', markersize=5)
ax.set_title("theta_1, theta_3, theta_4")
ax.grid(True)

ax = fig.add_subplot(2, 2, 3, projection='3d')
ax.plot3D(x_train[:npr,0], x_train[:npr,2], x_train[:npr,4], 'bo')
ax.plot3D(x_train_pred[0:npr,0], x_train_pred[0:npr,2], x_train_pred[0:npr,4], 'go', label='Projection')
# ax.plot3D(x_path[:,0], x_path[:,2], x_path[:,4], 'mo-', label='Projection', markersize=5)
ax.set_title("theta_1, theta_3, theta_5")
ax.grid(True)

ax = fig.add_subplot(2, 2, 4, projection='3d')
ax.plot3D(x_train[:npr,1], x_train[:npr,3], x_train[:npr,4], 'bo')
ax.plot3D(x_train_pred[0:npr,1], x_train_pred[0:npr,3], x_train_pred[0:npr,4], 'go', label='Projection')
# ax.plot3D(x_path[:,1], x_path[:,3], x_path[:,4], 'mo-', label='Projection', markersize=5)
ax.set_title("theta_2, theta_4, theta_5")
ax.grid(True)

plt.figure(2)
ax = plt.axes(projection='3d')
# ax.plot3D(x_train[:npr,0], x_train[:npr,1], x_train[:npr,3], 'k.')
ax.plot3D(x_test[:npr,0], x_test[:npr,1], x_test[:npr,3], 'bo')
ax.plot3D(x_test_pred[0:npr,0], x_test_pred[0:npr,1], x_test_pred[0:npr,3], 'go', label='Projection')
# ax.plot3D(x_path[:,0], x_path[:,1], x_path[:,3], 'mo-', label='Projection', markersize=10)
# ax.plot3D(x_path[:1,0], x_path[:1,1], x_path[:1,3], 'rp-', label='Projection', markersize=10)
# ax.scatter(x_path[-1,0], x_path[-1,1], x_path[-1,3], 'gp-', label='Projection', markersize=14)

plt.figure(3)
ax = plt.axes(projection='3d')
ax.plot3D(z_train_pred[:npr,0], z_train_pred[:npr,1], z_train_pred[:npr,3], 'bo')
ax.plot3D(z_test_pred[:npr,0], z_test_pred[:npr,1], z_test_pred[:npr,3], 'go')
# plt.plot(z_path[:,0],z_path[:,1], 'ko-')

plt.figure(5)
plt.hist(z_train_pred[:,0], bins=30)

plt.show()