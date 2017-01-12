import tensorflow as tf
import matplotlib.pyplot as plt
from loadData import *
import sys
from model import model0
sys.path.append("/Users/matt/misc/tfFunctions/")

def varSum(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

def lossFn(y,yPred):
    return tf.reduce_mean(tf.square(tf.sub(y,yPred)))

def trainer(lossFn, learningRate):
    return tf.train.AdamOptimizer(learningRate).minimize(lossFn)

def nodes(batchSize,inSize,outSize,trainOrTest):
    if trainOrTest == "train":
        csvPath = "train.csv"
        print("Training")
        is_training = True
        shuffle = True
    elif trainOrTest == "test":
        print("Testing")
        csvPath = "testCV.csv"
        is_training = False
        shuffle = False
    X,Y,path = read(csvPath=csvPath,
            batchSize=batchSize,
            inSize=inSize,
            outSize=outSize,
            shuffle=shuffle) #nodes
    return X,Y,path

if __name__ == "__main__":
    import pdb
    inSize = [256,256]
    outSize = [32,32]
    batchSize = 10

    print("Input Dim = h={0},w={1}".format(inSize,outSize))
    is_training = tf.placeholder(tf.bool)
    X,Y, path = nodes(batchSize=batchSize,inSize=inSize,outSize=outSize,trainOrTest="train")
    YPred = model0(X,is_training=is_training)
    mse = lossFn(Y,YPred)
    nEpochs = 10
    varSum(mse)
    load = 1
    trainOp = trainer(mse,0.001)

    merged = tf.summary.merge_all()
    saver = tf.train.Saver()
    savePath = "models/model0.tf"

    count = 0
    for epoch in xrange(nEpochs):

        if epoch > 0:
            load = 1
        with tf.Session() as sess:
            if load == 1:
                saver.restore(sess,savePath)
            else:
                tf.initialize_all_variables().run()
            tf.local_variables_initializer().run()
            train_writer = tf.summary.FileWriter("summary/train/",sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            try:
                while True:
                    _, summary = sess.run([trainOp,merged],feed_dict={is_training:True})
                    count += batchSize
                    train_writer.add_summary(summary,count)
                    if coord.should_stop():
                        break
            except Exception,e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            saver.save(sess,savePath)
            sess.close()
