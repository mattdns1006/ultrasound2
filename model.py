import tensorflow as tf
import numpy as np
import sys
sys.path.append("/Users/matt/misc/tfFunctions")
from layers import *
import tensorflow.contrib.layers as layers
from batchNorm2 import bn

#bn = layers.batch_norm
fmp = tf.nn.fractional_max_pool
mp = tf.nn.max_pool
conv = tf.nn.conv2d
dilconv = tf.nn.atrous_conv2d
convUp = tf.nn.conv2d_transpose
af = tf.nn.relu
W = weightVar
B = biasVar

def convolution2d(inTensor,size,inFeats,outFeats):
    with tf.name_scope("conv2d"):
        with tf.name_scope("weights"):
            weight = W([size,size,inFeats,outFeats])
        with tf.name_scope("biases"):
            bias = B([outFeats])
        with tf.name_scope("conv"):
            filtered = conv(inTensor,weight,strides=[1,1,1,1],padding='SAME') + bias
    return filtered

def dilated_convolution2d(inTensor,fS,inFeats,outFeats,dilation):
    with tf.name_scope("weights"):
        weight = W([fS,fS,inFeats,outFeats])
    with tf.name_scope("biases"):
        bias = B([outFeats])
    with tf.name_scope("conv"):
        out = tf.nn.atrous_conv2d(inTensor,weight,dilation,padding='VALID') + bias
    return out 

def denseBlock(x,inFeats,outFeats,layerNo,is_training):
    def growth(inTensor,toAdd,inFeats,outFeats,growthNumber,layerNo):
        with tf.name_scope("growth_{0}".format(growthNumber)):
            filtered = convolution2d(inTensor,1,inFeats,outFeats)
            with tf.name_scope("add"):
                out = af(bn(toAdd + filtered,is_training=is_training,name="bn_{0}_{1}".format(growthNumber,layerNo)))
        return out
    x1 = growth(inTensor=x,toAdd=x,inFeats=inFeats,outFeats=outFeats,growthNumber=1,layerNo=layerNo)
    x10 = tf.add(x1,x)
    x2 = growth(inTensor=x1,toAdd=x10,inFeats=inFeats,outFeats=outFeats,growthNumber=2,layerNo=layerNo)
    x210 = tf.add(x2,x10)
    x3 = growth(inTensor=x2,toAdd=x210,inFeats=inFeats,outFeats=outFeats,growthNumber=3,layerNo=layerNo)
    #x3210 = tf.add(x3,x210)
    #x4 = growth(inTensor=x3,toAdd=x3210,inFeats=inFeats,outFeats=outFeats,growthNumber=4,layerNo=layerNo)
    return x3

def denseNet(x,is_training,filterSize=3,initFeats=16):
    X = convolution2d(x,filterSize,1,initFeats)
    X = af(bn(X,is_training=is_training,name="bn_conv_0"))
    for layerNo in range(6):
        with tf.variable_scope("dense_{0}".format(layerNo)):
            X = denseBlock(X,initFeats,initFeats,layerNo,is_training)
            print(X.get_shape().as_list())
            holes = np.power(2,layerNo)
            print(holes)
            X = dilated_convolution2d(X,3,initFeats,initFeats,3)
            print(X.get_shape().as_list())
            X = af(bn(X,is_training=is_training,name="bn_conv"))
            print(X.get_shape().as_list())

    X = convolution2d(X,filterSize,initFeats,1)
    X = af(bn(X,is_training=is_training,name="bn_conv_out"))
    return X

def model1(x,is_training,filterSize=3,initFeats=16):
    X = convolution2d(x,filterSize,1,initFeats)
    X = af(bn(X,is_training=is_training,name="bn_conv_init"))
    for layerNo in range(6):
        X = convolution2d(X,filterSize,initFeats,initFeats)
        X = af(bn(X,is_training=is_training,name="bn_conv_{0}".format(layerNo)))
        X = dilated_convolution2d(X,3,initFeats,initFeats,3)
    X = convolution2d(X,filterSize,initFeats,1)
    X = af(bn(X,is_training=is_training,name="bn_conv_out"))
    return X

def model0(x,is_training,nLayers=4,initFeats=16,filterSize=3):
    X = convolution2d(x,filterSize,1,initFeats)
    X = bn(X,is_training,name="bn")
    X = af(X)
    for layerNo in range(nLayers):
        if layerNo == 0:
            inFeats = initFeats 
            outFeats = initFeats 
        elif layerNo == nLayers- 1:
            outFeats = 1
        else:
            inFeats = outFeats 
            outFeats = initFeats
        with tf.variable_scope("layerNo_{0}".format(layerNo)):
            X = convolution2d(X,filterSize,inFeats,outFeats)
            X = bn(X,is_training,name="bn_{0}".format(layerNo))
            X = af(X)
            if layerNo == nLayers- 1:
                break
            X = tf.nn.max_pool(X,[1,3,3,1],[1,2,2,1],'SAME')
    return X

if __name__ == "__main__":
    import pdb
    import numpy as np
    X = tf.placeholder(tf.float32,shape=[None,128,128,1])
    is_training = tf.placeholder(tf.bool,name="is_training")
    Y = denseNet(X,is_training=is_training,initFeats=16)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(10):
            x = np.random.rand(1,128,128,1)
            y_ = sess.run([Y],feed_dict={X:x,is_training.name:True})
            print(y_[0].shape)
            if i == 9:
                pdb.set_trace()


