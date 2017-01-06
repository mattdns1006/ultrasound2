import tensorflow as tf
import sys
sys.path.append("/Users/matt/misc/tfFunctions")
from layers import *
import tensorflow.contrib.layers as layers

bn = layers.batch_norm
fmp = tf.nn.fractional_max_pool
mp = tf.nn.max_pool
conv = tf.nn.conv2d
dilconv = tf.nn.atrous_conv2d
convUp = tf.nn.conv2d_transpose
af = tf.nn.relu
W = weightVar
B = biasVar

def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def block(X,inFeats,outFeats,is_training,layerNo,res):
    fS = 3
    with tf.name_scope(layerNo):
        with tf.name_scope("weights"):
            weight = W([fS,fS,inFeats,outFeats])
            variable_summaries(weight)
        with tf.name_scope("biases"):
            bias = B([outFeats])
            variable_summaries(bias)
        with tf.name_scope("conv"):
            filtered = conv(X,weight,strides=[1,1,1,1],padding='SAME') + bias
            #filtered = dilconv(X,weight,rate=3,padding='SAME') + bias
            variable_summaries(filtered)
        with tf.name_scope("bn"):
            normalized = bn(filtered,is_training=is_training)
            variable_summaries(normalized)
        with tf.name_scope("activation"):
            activation = af(normalized)
            variable_summaries(activation)
        if res == True:
            res = af(X + activation)
            return res
        else:
            return activation

def model(X,is_training):
    for i in range(5):
        if i == 0:
            inFeats = 1
            outFeats = 16 
            inTensor = X
            res = False
        else:
            inFeats = outFeats
            outFeats = 16 
            inTensor = outTensor
            res = True
        outTensor = block(inTensor,inFeats=inFeats,outFeats=outFeats,is_training=True,layerNo="{0}".format(i),res=res)
    return X

if __name__ == "__main__":
    import numpy as np
    import pdb
    X = tf.placeholder(tf.float32,shape=[None,64,64,1])
    Y = model(X,is_training=True)
    merged = tf.summary.merge_all()
    with tf.Session() as sess:
        train_writer = tf.train.SummaryWriter("summary",sess.graph)
        tf.global_variables_initializer().run()
        x = np.random.rand(1,64,64,1)
        summary,_ = sess.run([merged,Y],feed_dict={X:x})
        train_writer.add_summary(summary)


