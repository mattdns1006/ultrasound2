import tensorflow as tf
from loadData import *
import sys
from model import model0, model1, denseNet
sys.path.append("/Users/matt/misc/tfFunctions/")
from dice import dice as dice
import cv2

def varSum(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)

def imgSum(img):
    tf.summary.image("Image",img)

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
    is_training = tf.placeholder(tf.bool)
    with tf.variable_scope("Input"):
        imgSum(X)
    with tf.variable_scope("Truth"):
        imgSum(Y)
    with tf.variable_scope("Prediction"):
        YPred = model1(X,is_training=is_training,initFeats=16)
        #YPred = model0(X,is_training=is_training,nLayers=3)
        imgSum(YPred)
    with tf.variable_scope("MSE"):
        mse = lossFn(Y,YPred)
        varSum(mse)
    with tf.variable_scope("Dice"):
        diceScore, diceThresh = dice(YPred,Y)
        imgSum(diceThresh)
        varSum(diceScore)
    trainOp = trainer(mse,0.0001)
    return X,Y,YPred,path,mse,diceScore,is_training,trainOp

if __name__ == "__main__":
    import pdb
    nEpochs = 10

    flags = tf.app.flags
    FLAGS = flags.FLAGS 
    flags.DEFINE_float("lr",0.001,"Initial learning rate.")
    flags.DEFINE_integer("sf",256,"Size of input image")
    flags.DEFINE_integer("initFeats",8,"Initial number of features.")
    flags.DEFINE_integer("incFeats",16,"Number of features growing.")
    flags.DEFINE_integer("nDown",8,"Number of blocks going down.")
    flags.DEFINE_integer("bS",3,"Batch size.")
    flags.DEFINE_integer("load",1,"Load saved model.")
    inSize = [96,96]
    outSize = [60,60]
    savePath = "models/model0.tf"

    count = load = 0
    for epoch in range(nEpochs):
        print("{0} of {1}".format(epoch,nEpochs))
        if epoch > 0:
            load = 1 
        tf.reset_default_graph()
        X,Y,YPred,path,mse,diceScore,is_training,trainOp = nodes(batchSize=FLAGS.bS,inSize=inSize,outSize=outSize,trainOrTest="train")
        saver = tf.train.Saver()
        merged = tf.summary.merge_all()
        with tf.Session() as sess:
            if FLAGS.load == 1 or load == 1:
                print("Restoring")
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
                    count += FLAGS.bS
                    train_writer.add_summary(summary,count)
                    if coord.should_stop():
                        break
            except Exception as e:
                coord.request_stop(e)
            finally:
                coord.request_stop()
                coord.join(threads)
            saver.save(sess,savePath)
            sess.close()
