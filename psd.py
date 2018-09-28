import tensorflow as tf

def encoder(Y,encoder_size,scope='encoder'):
    with tf.variable_scope(scope):
        G=tf.Variable(tf.random_normal(shape=[encoder_size,encoder_size],stddev=0.01))
        W=tf.Variable(tf.random_normal(shape=[Y.shape[1].value,encoder_size],stddev=0.01))
        D=tf.Variable(tf.zeros(encoder_size))
        return tf.matmul(tf.nn.tanh(tf.add(tf.matmul(Y,W),D)),G)

def loss(Y,Z,F,B,alpha,phi,scope='loss'):
    loss1=tf.sqrt(tf.nn.l2_loss(tf.subtract(Y,tf.matmul(Z,B))))
    loss2=tf.reduce_mean(tf.abs(Z),axis=-1)
    loss3=tf.sqrt(tf.nn.l2_loss(tf.subtract(Z,F)))
    with tf.variable_scope(scope):
        loss=loss1+phi*loss2+alpha*loss3
    return tf.reduce_mean(loss)
