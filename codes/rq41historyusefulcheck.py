import glob
import os
import sys
print(sys.path)
import numpy as np
import tensorflow as tf
import collections
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from rq41publicfuncs import  modeldataprepare, readsortdata, preprocessingdata, \
                                               WGstruct, modelhistonly, \
                                               datasplit2seg, compile_and_fit,paramtunemetricsplot,cmsave

seed = 42
tf.random.set_seed(seed)


def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        filename = os.path.basename(files)[:-4]
        df = readsortdata(files)

        temp_train_df, temp_val_df, temp_test_df = datasplit2seg(df, 0.9)

        train_df, val_df, test_df = preprocessingdata(temp_train_df, temp_val_df, temp_test_df)
        # plotlasslabel(train_df)
        counter = collections.Counter(train_df['bug'])
        neg = counter[0]
        pos = counter[1]
        print(neg, pos, train_df.shape[0])

        histcolumns = {name: i for i, name in enumerate(df.columns)}

        input_widths = 60
        label_widths = 1
        shiftwidth = 0

        lr = 0.001
        stride = 1
        batch = 32
        regulazer = 0.0005
        dropout = 0.5
        activation = tf.keras.layers.LeakyReLU(alpha=0.003)
        epoch = 500


        w = WGstruct(input_width=input_widths, label_width=label_widths, shift=shiftwidth,
                            sequence_stride=stride, batch_size=batch,
                            train_df=train_df, val_df=val_df, test_df=test_df,
                            label_columns=['bug'])

        # print(w)
        hist_train, curr_train, label_train = modeldataprepare(w.train)
        hist_val, curr_val, label_val = modeldataprepare(w.val)
        hist_test, curr_test, label_test = modeldataprepare(w.test)
        print(np.array(hist_train).shape, np.array(label_train).shape)
        print(np.array(hist_val).shape, np.array(label_val).shape)

        cmlist = []
        for i in range(50):
            checkpoint_path = os.path.join(argv[2], '{}_{}.{}'.format(filename, i, 'ckpt'))
            model = modelhistonly(input_widths - label_widths, len(histcolumns), np.log([pos / neg]), dropout, regulazer, activation)
            # model, batch, maxepochs, patience, lr, hist_train, label_train, hist_val, label_val
            history = compile_and_fit(model, batch, epoch, checkpoint_path, lr, hist_train, label_train, hist_test, label_test)

            # Re-evaluate the model:
            loss_val, acc_val, pre_val, rec_val, auc_val, prc_val = model.evaluate(
                {"histfea": hist_test},
                {"bugprone": label_test},
                batch_size=batch, verbose=2)
            if 0 == (pre_val + rec_val):
                f1=0
            else:
                f1 = 2 * (pre_val * rec_val) / (pre_val + rec_val)
            dict = {'dataset':filename, 'index': i, 'valloss': loss_val, 'valacc': acc_val, 'valpre': pre_val, 'valrec': rec_val,
                    'valauc': auc_val,
                    'valprc': prc_val, 'valf1': f1}

            # y_pred = model.predict({"histfea": hist_test})
            # y_true = getreallabels(w.test)
            # print(np.array(y_pred).shape, y_true.shape)
            #
            #
            # dict["test_auc"] = metrics.roc_auc_score(y_true, y_pred)
            # y_pred = classify(np.array(y_pred).reshape(-1, 1))
            # dict["test_f1"] = metrics.f1_score(y_true, y_pred, average='macro')
            #
            # cm = confusion_matrix(y_true, y_pred)
            #
            # dict["test_cm"] = cm
            cmlist.append(dict)
            paramtunemetricsplot(argv[2], filename, i, history)

        cmsave(cmlist, argv[3], "hist50", filename)



if __name__ == '__main__':
    main(sys.argv)











