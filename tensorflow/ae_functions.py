
import tensorflow as tf
import numpy as np

def log2path(X):
    f = open('../paths/path.txt','w')
    f.write(str(X.shape[0]) + '\n')

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            f.write(str(X[i,j])+ ' ')
        f.write('\n')

    f.close()

def dataINnOUT(Xi, Xo, Z):
    f = open('./inout_5bars.txt','w')

    for i in range(Xi.shape[0]):
        for j in range(Xi.shape[1]):
            f.write(str(Xi[i,j])+ ' ')
        for j in range(Z.shape[1]):
            f.write(str(Z[i,j])+ ' ')
        for j in range(Xo.shape[1]):
            f.write(str(Xo[i,j])+ ' ')
        f.write('\n')

    f.close()

# Export net to text file - this function must be called within the session
def export_net(W, B, x_max, x_min, sess):
    f = open('./net_abb3.netxt','w')

    k = W.keys()
    n = int(len(k)/2)
    f.write(str(2*n) + ' ')
    
    # W's in encoder
    for i in range(n):
        sth = 'encoder_h' + str(i+1)
        w = sess.run(W[sth])
        f.write(str(w.shape[0]) + ' ' + str(w.shape[1]) + ' ')
        for j in range(w.shape[0]):
            for k in range(w.shape[1]):
                f.write(str(w[j,k]) + ' ')
    
    # W's in decoder
    for i in range(n):
        sth = 'decoder_h' + str(i+1)
        w = sess.run(W[sth])
        f.write(str(w.shape[0]) + ' ' + str(w.shape[1]) + ' ')
        for j in range(w.shape[0]):
            for k in range(w.shape[1]):
                f.write(str(w[j,k]) + ' ')

    # b's in encoder
    for i in range(n):
        sth = 'encoder_b' + str(i+1)
        b = sess.run(B[sth])
        f.write(str(b.shape[0]) + ' ')
        for j in range(b.shape[0]):
                f.write(str(b[j]) + ' ')

    # b's in decoder
    for i in range(n):
        sth = 'decoder_b' + str(i+1)
        b = sess.run(B[sth])
        f.write(str(b.shape[0]) + ' ')
        for j in range(b.shape[0]):
                f.write(str(b[j]) + ' ')

    f.write(str(x_max) + ' ' + str(x_min))

    f.close()

def normz(x, x_max, x_min):
    # for i in range(4):
    #     x[:,i] = (x[:,i]-x_min[i])/(x_max[i]-x_min[i])
    
    x = (x-x_min)/(x_max-x_min)

    return x

def denormz(x, x_max, x_min):
    # for i in range(4):
    #     x[:,i] = x[:,i]*(x_max[i]-x_min[i]) + x_min[i]
    x = x*(x_max-x_min) + x_min
    
    return x

# -----------------------------------------------------------------------

def next_batch(num, data):
    '''
    Return a total of `num` random samples. 
    Similar to mnist.train.next_batch(num)
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]

    return np.asarray(data_shuffle)


# Build weight and bias matrices
def wNb(num_input, hidden_layers):
    weights = {}
    biases = {}
    h = hidden_layers
    h = np.insert(h, 0, num_input)

    # initializer = tf.contrib.layers.xavier_initializer()
    # w1 = tf.Variable(initializer(w1_shape))
    # b1 = tf.Varialbe(initializer(b1_shape))

    # Encoder
    for i in range(len(h)-1):
        sth = 'encoder_h' + str(i+1)
        weights.update({sth: tf.Variable(tf.random_normal([h[i], h[i+1]]))})
        stb = 'encoder_b' + str(i+1)
        biases.update({stb: tf.Variable(tf.random_normal([h[i+1]]))})
    
    # Decoder
    h = np.flipud(h)
    for i in range(len(h)-1):
        sth = 'decoder_h' + str(i+1)
        weights.update({sth: tf.Variable(tf.random_normal([h[i], h[i+1]]))})
        stb = 'decoder_b' + str(i+1)
        biases.update({stb: tf.Variable(tf.random_normal([h[i+1]]))})
    
    return weights, biases

def activF(x, activation_index):
    if activation_index==1:
        return tf.nn.sigmoid(x)
    if activation_index==2:
        return tf.nn.relu(x)
    if activation_index==3:
        return tf.nn.tanh(x)


# Building the encoder
def encoder(x, weights, biases, activation_index=1):
    # First hidden fully connected layer 
    layer = activF(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']), activation_index)

    # Remaining hidden fully connected layer 
    for i in range(1, int(len(weights)/2)):
        sth = 'encoder_h' + str(i+1)
        stb = 'encoder_b' + str(i+1)
        layer = activF(tf.add(tf.matmul(layer, weights[sth]), biases[stb]), activation_index)

    return layer

# Building the decoder
def decoder(x, weights, biases, activation_index=1):
    # First hidden fully connected layer 
    layer = activF(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']), activation_index)

    for i in range(1, int(len(weights)/2)):
        sth = 'decoder_h' + str(i+1)
        stb = 'decoder_b' + str(i+1)
        layer = activF(tf.add(tf.matmul(layer, weights[sth]), biases[stb]), activation_index)

    return layer
