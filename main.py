import tensorflow as tf
import matplotlib.pyplot as plt
from loadData import *
import sys
sys.path.append("/Users/matt/misc/tfFunctions/")




def nodes(trainOrTest):
    if trainOrTest == "train":
        csvPath = "trainCV.csv"
        print("Training")
        is_training = True
        shuffle = True
    elif trainOrTest == "test":
        print("Testing")
        csvPath = "testCV.csv"
        is_training = False
        shuffle = False
    X,Y,path = read(batchSize,shuffle=shuffle) #nodes

if __name__ == "__main__":
    pass
