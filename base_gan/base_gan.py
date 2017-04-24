# import necessary packages
import tensorflow as tf
import os
import numpy as np
from keras.utils import np_utils
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


# download/load tensorflow data 
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)


# 

n_epoch = 3000
batch_size = 64
h_dim = 128
z_dim = 100
n_sample = mnist.train.images.shape[0]


# Util functions definition

def flatten(x):
    out_dim = np.prod(x.get_shape()[1:].as_list())
    return tf.reshape(x, shape=(-1, out_dim))

def weight(shape):
    with tf.variable_scope('weight'):
        return tf.get_variable('weight', shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

def bias(shape):
    with tf.variable_scope('bias'):
        init = np.ones(shape, dtype=np.float32) * 0.1
        return tf.get_variable('bias', initializer=init, dtype=tf.float32)

def fc_layer(x, unit, tag=None, name=None, activation=None):
    with tf.variable_scope(name):
        in_dim = int(x.get_shape()[-1])
        out_dim = unit
        w = weight([in_dim, out_dim])
        b = bias([out_dim])
        if tag:
            tf.add_to_collection(name=tag, value=w)
            tf.add_to_collection(name=tag, value=b)
        out = tf.matmul(x, w) + b
        return activation(out) if activation else out
    
# As `mnist` dataset provided by `Tensorlfow` is # normalized (subtract by `mean` and divided by `std`)
# we need reversed option for normalization
# reference https://keras-cn.readthedocs.io/en/latest/blog/cnn_see_world/
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# save images to logging directory  
def save_image(G, epoch, n_imgs=20, img_size=(128, 128)):
    if not os.path.exists('out/epoch_{0}'.format(epoch)):
        os.mkdir('out/epoch_{0}'.format(epoch))

    imgs = G.eval(feed_dict={zs:sampler((n_imgs, z_dim))})
    for step, img in enumerate(imgs):
        img = deprocess_image(np.squeeze(img))
        img = Image.fromarray(img)
        img.resize(img_size).save('out/epoch_{0}/{1}.png'.format(epoch, step))
     
# show images in Jupyter Notebook   
def show_image(G, n_imgs=5):
    for step in range(n_imgs):
        plt.figure()
        img = G.eval(feed_dict={zs:sampler((1, z_dim))})
        img = deprocess_image(np.squeeze(img))
        plt.imshow(img, cmap='Greys_r')
        plt.show()


# Discriminator is a simple 3 layers BP network

def discriminator(x, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        tag = 'D' if reuse else None
        x = flatten(x)
        x = fc_layer(x, 128, activation=tf.nn.relu, name='d_fc1', tag=tag)
        x = fc_layer(x, 1, name='d_fc3', tag=tag)

        return tf.nn.sigmoid(x), x

def generator(z, name):
    with tf.variable_scope(name):
        x = fc_layer(z, 128, activation=tf.nn.relu, name='g_fc1', tag='G')
        x = fc_layer(x, 784, activation=tf.nn.sigmoid, name='g_fc3', tag='G')

        return tf.reshape(x, [-1, 28, 28, 1])



# uniform random generator
def sampler(shape, method='unif'):
    if method == 'unif':
        return np.random.uniform(-1., 1., size=shape)
    return np.random.random(shape)


# plot 16 images in one Image
def plot(samples):
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


# Model building
xs = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name='input')
ys = tf.placeholder(tf.float32, shape=(None, 10), name='label')
zs = tf.placeholder(tf.float32, shape=[None, z_dim], name='noise')

G = generator(zs, name='generator')
D_real, D_logit = discriminator(xs, name='discriminator')
D_fake, _D_logit = discriminator(G, name='discriminator', reuse=True)

# Loss definition
real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(D_logit), logits=D_logit))
fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(_D_logit), logits=_D_logit))

d_loss = real_loss + fake_loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(_D_logit), logits=_D_logit))
d_optim = 

# Optimizer: Adam with learning rate=2e-4
tf.train.AdamOptimizer(2e-4).minimize(d_loss, var_list=tf.get_collection('D'))
g_optim = tf.train.AdamOptimizer(2e-4).minimize(g_loss, var_list=tf.get_collection('G'))


# Tensorflow variables initialization
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


#  Training steps
d_step = 1
g_step = 2

for epoch in range(n_epoch):
    for i in range(n_sample / batch_size):
        X_train, y_train = mnist.train.next_batch(batch_size)
        X_train = X_train.reshape((-1, 28, 28, 1))
        noise = sampler((batch_size, z_dim))
        
        for i in range(d_step):
            _, d_loss_out = sess.run([d_optim, d_loss], feed_dict={xs:X_train, zs:noise})
            
        for i in range(g_step):
            _, g_loss_out = sess.run([g_optim, g_loss], feed_dict={zs:noise})
        
    print  d_loss_out, g_loss_out
    if epoch % 5 == 0:
        #save_image(G, epoch)
        show_image(G)