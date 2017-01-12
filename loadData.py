import tensorflow as tf
import cv2, glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import pdb

def makeCsv():
    y = glob.glob("train/*_mask.jpg")
    x = [i.replace("_mask","") for i in y]
    pdb.set_trace()
    csv = pd.DataFrame({"x":x,"y":y})
    csv.to_csv("train.csv",index=0)

def show(img):
    plt.imshow(img,cmap=cm.gray)
    plt.show()

def showBatch(batchX,batchY,figsize=(15,15)):
    n, h, w, c = batchX.shape
    batchX = batchX.reshape(n*h,w)
    n, h, w, c = batchY.shape
    batchY = batchY.reshape(n*h,w)
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    plt.imshow(batchX,cmap=cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(batchY,cmap=cm.gray)
    plt.show()

def getImg(path,size):
    imageBytes = tf.read_file(path)
    decodedImg = tf.image.decode_jpeg(imageBytes)
    decodedImg = tf.image.resize_images(decodedImg,size)
    decodedImg = tf.cast(decodedImg,tf.float32)
    decodedImg = tf.mul(decodedImg,1/255.0)
    return decodedImg

def read(csvPath,batchSize,inSize,outSize,shuffle):
    csv = tf.train.string_input_producer([csvPath],num_epochs=1,shuffle=shuffle)
    reader = tf.TextLineReader(skip_header_lines=1)
    k, v = reader.read(csv)
    defaults = [tf.constant([],dtype = tf.string),
                tf.constant([],dtype = tf.string)]
    xPath, yPath = tf.decode_csv(v,record_defaults = defaults)
    xPathRe = tf.reshape(xPath,[1])

    x,y = getImg(xPath,inSize), getImg(yPath,outSize)
    inSize += [1]
    outSize += [1]
    Q = tf.FIFOQueue(64,[tf.float32,tf.float32,tf.string],shapes=[inSize,outSize,[1]])
    enQ = Q.enqueue([x,y,xPathRe])
    QR = tf.train.QueueRunner(
            Q,
            [enQ]*8,
            Q.close(),
            Q.close(cancel_pending_enqueues=True)
            )
    tf.train.add_queue_runner(QR) 
    dQ = Q.dequeue()
    X,Y,path = tf.train.batch(dQ,batchSize,16)
    return X, Y, path

if __name__ == "__main__":
    inSize = [200,200]
    outSize = [100,100]
    X, Y, path = read(csvPath="train.csv",batchSize=4,inSize=inSize,outSize=outSize,shuffle=True)
    init_op = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init_op)
        tf.initialize_local_variables().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        count = 0
        try:
            while True:
                x, y, path_ = sess.run([X,Y,path])
                showBatch(x,y)
                count += x.shape[0]
                pdb.set_trace()
                if coord.should_stop():
                    break
        except Exception,e:
            coord.request_stop(e)
        finally:
            coord.request_stop()
            coord.join(threads)
