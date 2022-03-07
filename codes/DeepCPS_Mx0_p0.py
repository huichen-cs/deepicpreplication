# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys
import pandas as pd
import collections
import numpy as np
import tensorflow as tf
from rq43publicfuncs import  modeldataprepare, WGstruct, modelcombinefeature, compilefitsavemodel,modelloadandpred,\
                                               saverettocsv
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\balanceddata"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio4"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\DeepCPS_Mx0_p0\train"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\DeepCPS_Mx0_p0\test"
def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        segnum = int(os.path.basename(files)[-5])
        if (filename != "col"):
            continue

        if (segnum != 2):
            continue
        print(segnum)
        train_df = pd.read_csv(files)
        counter = collections.Counter(train_df['bug'])
        neg = counter[0]
        pos = counter[1]

        histcolumns = {name: i for i, name in enumerate(train_df.columns)}
        currcolumns = {name: i for i, name in enumerate(train_df.columns) if name != 'bug'}
        print(histcolumns)

        input_widths = 11
        label_widths = 1
        shiftwidth = 0
# tune lr/lr schedule(log), dropout rate[hist, curr], regulazer:  lan,cam,der,mat,col
        lr = 0.001
        stride = 1
        batch = 32
        regulazer = 0.00005
        dropouthist = 0.5
        dropoutcurr = 0.5
        activation = tf.keras.layers.LeakyReLU(alpha=0.003)
        epoch = 200

        rets = []

        val_df = train_df.loc[0.9*len(train_df):, :]
        w = WGstruct(input_width=input_widths, label_width=label_widths, shift=shiftwidth,
                     sequence_stride=stride, batch_size=batch,
                     train_df=train_df.loc[0:0.9*len(train_df), :], val_df=val_df, test_df=val_df,
                     label_columns=['bug'])

        # print(w)
        hist_train, curr_train, label_train = modeldataprepare(w.train)
        hist_val, curr_val, label_val = modeldataprepare(w.val)

        # compile and fit model with balanced train data
        checkpoint_path = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, segnum, "Mx0_p0_train", 'ckpt'))
        modelpath = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, segnum, "Mx0_p0_train", 'h5'))
        predicpath = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, segnum, "Mx0_p0_train", 'csv'))
        plotpath = argv[3]

        model = modelcombinefeature(input_widths - label_widths, label_widths, len(histcolumns), len(currcolumns),
                                    np.log([pos / neg]), dropouthist, dropoutcurr, regulazer, activation)
        model = compilefitsavemodel(model, filename, segnum, "Mx0_p0_train", epoch, batch, lr,
                                    hist_train, curr_train, label_train,
                                    hist_val, curr_val, label_val,
                                    checkpoint_path, modelpath, predicpath, plotpath)

        for i in range(4):
            test_path = os.path.join(argv[2], '{}_{}_{}_{}.{}'.format(filename, 'seg3', 'subseg', i, 'csv'))
            test_df = pd.read_csv(test_path)
            test_df = test_df.drop(columns=['Unnamed: 0',"commitdate"])
            print(test_df.columns)
            wtest = WGstruct(input_width=input_widths, label_width=label_widths, shift=shiftwidth,
                             sequence_stride=stride, batch_size=batch,
                             train_df=test_df, val_df=test_df,
                             test_df=test_df,
                             label_columns=['bug'])
            hist_val_real, curr_val_rea, label_val_real = modeldataprepare(wtest.val)

            predicpath = os.path.join(argv[4], '{}_{}_{}.{}'.format(filename, i, "Mx0_p0_test", 'csv'))
            dict = modelloadandpred(model, i, filename, hist_val_real, curr_val_rea, label_val_real, predicpath)

            rets.append(dict)

        path = os.path.join(argv[4], '{}_{}_{}.{}'.format(filename, 'seg2train', 'subsegin3test', 'csv'))
        saverettocsv(rets, path)


if __name__ == '__main__':
    main(sys.argv)

