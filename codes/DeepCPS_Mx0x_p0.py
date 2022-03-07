# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys
import pandas as pd
import collections
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from rq43publicfuncs import  modeldataprepare, WGstruct, modelcombinefeature, compilefittunesavemodel,modelloadandpred,\
                                               saverettocsv
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\balanceddata"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio4SMOTEPC"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio4"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\DeepCPS_Mx0x_p0\train"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\DeepCPS_Mx0x_p0\test"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    tarin_data_x0_path = argv[1]
    tarin_data_x_path = argv[2]
    test_data_path = argv[3]
    train_save_path = argv[4]
    test_save_path = argv[5]

    datapath = os.path.join(tarin_data_x0_path, '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        segnum = int(os.path.basename(files)[-5])
        if (filename != "der") and (filename != "mat") and (filename != "lan") and (filename != "pig") and (filename != "bug") and (filename != "col"):
            continue

        if (segnum != 2):
            continue
        print(segnum)
        train_df_x0 = pd.read_csv(files)
        train_df_x_path = os.path.join(tarin_data_x_path, '{}_{}_{}_{}.{}'.format(filename, 'seg3', 'subseg', 0, 'csv'))
        train_df_x = pd.read_csv(train_df_x_path)
        train_df_x = train_df_x.drop(columns=['Unnamed: 0'])
        print(train_df_x0.shape, train_df_x.shape, len(train_df_x))
        print(train_df_x0.columns)

        frames = [train_df_x0, train_df_x]
        train_df = pd.concat(frames)
        print(len(train_df))
        counter = collections.Counter(train_df['bug'])
        neg = counter[0]
        pos = counter[1]

        histcolumns = {name: i for i, name in enumerate(train_df.columns)}
        currcolumns = {name: i for i, name in enumerate(train_df.columns) if name != 'bug'}
        print(histcolumns)

        input_widths = 11
        label_widths = 1
        shiftwidth = 0

        lr = 0.0005
        stride = 1
        batch = 32
        regulazer = 0.0005
        dropouthist = 0.5
        dropoutcurr = 0.5
        activation = tf.keras.layers.LeakyReLU(alpha=0.003)
        epoch = 200

        rets = []

        val_df = train_df.iloc[int(0.9 * len(train_df)):, :]
        w = WGstruct(input_width=input_widths, label_width=label_widths, shift=shiftwidth,
                     sequence_stride=stride, batch_size=batch,
                     train_df=train_df.iloc[0:int(0.9 * len(train_df)), :], val_df=val_df, test_df=val_df,
                     label_columns=['bug'])

        hist_train, curr_train, label_train = modeldataprepare(w.train)
        hist_val, curr_val, label_val = modeldataprepare(w.val)

        checkpoint_path = os.path.join(train_save_path, '{}_{}_{}.{}'.format(filename, segnum, "Mx0x_p0_train", 'ckpt'))
        modelpath = os.path.join(train_save_path, '{}_{}_{}.{}'.format(filename, segnum, "Mx0x_p0_train", 'h5'))
        predicpath = os.path.join(train_save_path, '{}_{}_{}.{}'.format(filename, segnum, "Mx0x_p0_train", 'csv'))
        plotpath = train_save_path

        model = modelcombinefeature(input_widths - label_widths, label_widths, len(histcolumns), len(currcolumns),
                                    np.log([pos / neg]), dropouthist, dropoutcurr, regulazer, activation)
        # checkpoint
        model = compilefittunesavemodel(model, filename, segnum, "Mx0x_p0_train", epoch, batch, lr,
                                    hist_train, curr_train, label_train,
                                    hist_val, curr_val, label_val,
                                    checkpoint_path, modelpath, predicpath, plotpath)
        model.load_weights(checkpoint_path)

        for i in range(1,4):
            test_path = os.path.join(test_data_path, '{}_{}_{}_{}.{}'.format(filename, 'seg3', 'subseg', i, 'csv'))
            test_df = pd.read_csv(test_path)
            test_df = test_df.drop(columns=['Unnamed: 0',"commitdate"])
            print(test_df.columns)
            wtest = WGstruct(input_width=input_widths, label_width=label_widths, shift=shiftwidth,
                             sequence_stride=stride, batch_size=batch,
                             train_df=test_df, val_df=test_df,
                             test_df=test_df,
                             label_columns=['bug'])
            hist_val_real, curr_val_rea, label_val_real = modeldataprepare(wtest.val)

            predicpath = os.path.join(test_save_path, '{}_{}_{}.{}'.format(filename, i, "Mx0x0_p_train", 'csv'))
            dict = modelloadandpred(model, i, filename, hist_val_real, curr_val_rea, label_val_real, predicpath)

            rets.append(dict)

        path = os.path.join(test_save_path, '{}_{}_{}.{}'.format(filename, 'seg2train', 'subsegin3test', 'csv'))
        saverettocsv(rets, path)


if __name__ == '__main__':
    main(sys.argv)

