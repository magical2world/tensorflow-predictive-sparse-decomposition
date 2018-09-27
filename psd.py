import tensorflow as tf

def encoder(Y,encoder_size,scope='encoder'):
    with tf.variable_scope(scope):
        G=tf.Variable(tf.random_normal(shape=[encoder_size,encoder_size],stddev=0.01))
        W=tf.Variable(tf.random_normal(shape=[encoder_size,Y.shape[1].value],stddev=0.01))
        D=tf.Variable(tf.zeros(encoder_size))
        return tf.matmul(G,tf.nn.tanh(tf.add(tf.multiply(W,Y),D)))

def loss(Y,Z,F,B,alpha,phi,scope='loss'):
    loss1=tf.nn.l2_loss(Y,tf.matmul(B,Z))
    loss2=tf.abs(Z)
    loss3=tf.nn.l2_loss(Z,F)
    with tf.variable_scope(scope):
        loss=loss1+phi*loss2+alpha*loss3
    return tf.reduce_mean(loss)
