import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
from psd import *

batch_size=128
image=tf.placeholder(tf.float32,[batch_size,784])

with tf.variable_scope('step1'):
    F=encoder(image,256)
with tf.variable_scope('step2'):
    Z=tf.Variable(tf.random_normal([batch_size,256],stddev=0.01))
with tf.variable_scope('step1'):
    B=tf.Variable(tf.random_normal([256,784],stddev=0.01))
total_loss=loss(image,Z,F,B,0.5,1)
vars=tf.trainable_variables()
step1_vars=[var for var in vars if 'step1' in var.name]
step2_vars=[var for var in vars if 'step2' in var.name]
restore_img=tf.matmul(Z,B)
global_step=tf.Variable(0, trainable=False,name='global_step',dtype=tf.int64)
lr=tf.train.exponential_decay(0.0001,global_step,decay_steps=100,decay_rate=0.95,staircase=True)
optimizer1=tf.train.AdamOptimizer(lr).minimize(total_loss,var_list=step1_vars,global_step=global_step)
optimizer2=tf.train.AdamOptimizer(lr).minimize(total_loss,var_list=step2_vars,global_step=global_step)

with tf.Session() as sess:
    import numpy as np
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        xs,ys=mnist.train.next_batch(batch_size)
        xs=(xs-0.5)*2
        l1,_=sess.run([total_loss,optimizer1],feed_dict={image:xs})
        l1,_=sess.run([total_loss,optimizer1],feed_dict={image:xs})
        l1,_=sess.run([total_loss,optimizer1],feed_dict={image:xs})
        l1, _ = sess.run([total_loss, optimizer1], feed_dict={image: xs})
        l2,_=sess.run([total_loss,optimizer2],feed_dict={image:xs})
        print('loss 1 is %f'%(l1))
        print('loss 2 is %f'%(l2))
        img=sess.run(restore_img,feed_dict={image:xs})
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(((img[1]+0.5)/2).reshape([28,28]))
    plt.subplot(122)
    plt.imshow(((xs[1]+0.5)/2).reshape([28,28]))
    plt.show()

    # print(xs.shape)