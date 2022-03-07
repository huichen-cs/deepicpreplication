import glob
import os
import sys
import pandas as pd
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from rq43publicfuncs import  modeldataprepare, WGstruct, modelcombinefeature, readsegdata, compilefitsavemodel,modelloadandpred,\
                                               paramtunemodelinitial, paramtunemodelupdate, saverettocsv, saveparamdictocsv

def main(argv):
    trainpath = argv[1]
    testpath = argv[2]
    trainsavepath = argv[3]
    testsavepath = argv[4]
    lr = np.random.uniform(0.001, 0.0000001)
    regulazer = np.random.uniform(0.01, 0.00001)
    dropouthist = np.random.random()
    dropoutcurr = np.random.random()
    batch = np.random.randint(20, 100)
    input_widths = np.random.randint(10, batch)
    j = np.random.randint(0, 300)
    print(lr, regulazer, dropouthist, dropoutcurr, batch, input_widths, j)
    return trainpath, testpath, trainsavepath, testsavepath, lr, regulazer, dropouthist, dropoutcurr, batch, input_widths, j



if __name__ == '__main__':
    main(sys.argv)

