import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)
import tensorflow as tf
from psd import *

batch_size=128
image=tf.placeholder(tf.float32,[batch_size,784])

F=encoder(image,256)
Z=tf.Variable(tf.zeros(256))
Z=tf.tile(Z,[batch_size,1])
print(Z)
B=tf.Variable(tf.random_normal([256,784],stddev=0.01))
total_loss=loss(image,Z,F,B,1,1)
optimizer=tf.train.AdamOptimizer(0.001).minimize(total_loss)

with tf.Session() as sess:
    for i in range(100):
        xs,ys=mnist.train.next_batch(batch_size)
        l,_=sess.run([total_loss,optimizer],feed_dict={image:xs})

    # print(xs.shape)