# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys
import pandas as pd
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf
from rq43publicfuncs import  saveretlr, saverettocsvlr

# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\balanceddata"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio4"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\LR_Mx0_p0\train"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\LR_Mx0_p0\test"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        print(filename)
        segnum = int(os.path.basename(files)[-5])
        # if (filename != "act") and (filename != "moz"):
        #     continue

        if (segnum != 2):
            continue
        print(segnum)
        train_df = pd.read_csv(files)

        rets = []
        y = train_df.pop('bug')
        x = train_df.to_numpy()

        # X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1)

        clf = LogisticRegression(random_state=0).fit(x, y)

        for i in range(4):
            test_path = os.path.join(argv[2], '{}_{}_{}_{}.{}'.format(filename, 'seg3', 'subseg', i, 'csv'))
            test_df = pd.read_csv(test_path)

            test_df = test_df.drop(columns=['Unnamed: 0',"commitdate"])
            test_y = test_df.pop('bug')
            test_x = test_df.to_numpy()
            pred_result = clf.predict(test_x)
            pred_prob_result = clf.predict_proba(test_x)

            predicpath = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, i, "LR_Mx0_p0_pred", 'csv'))
            saveretlr(pred_result, test_y.values.reshape(-1, 1), predicpath)

            predicpath = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, i, "LR_Mx0_p0_pred_prob", 'csv'))
            saveretlr(pred_prob_result, test_y.values.reshape(-1, 1), predicpath)
            # confusion metrix

            # acc = clf.score(test_x, test_y)

            aucf = tf.keras.metrics.AUC(name='auc')
            aucf.update_state(test_y.values.reshape(-1, 1), pred_prob_result[:, 1])
            auc = aucf.result().numpy()

            cm = confusion_matrix(test_y.values.reshape(-1, 1), pred_result)
            tn, fp, fn, tp = cm.ravel()

            recall = tp / (tp + fn)
            precision = tp / (tp + fp)
            acc = (tp + tn) / (tp + tn + fp + fn)
            f1 = 2 * (precision * recall) / (precision + recall)



            dict = {'cm':cm, 'valacc': acc, 'valpre': precision, 'valrec': recall,
                    'valauc': auc, 'f1': f1, 'filename': filename}


            #predicpath = os.path.join(argv[4], '{}_{}_{}.{}'.format(filename, i, "Mx0_p0_test", 'csv'))
            # print(pred_result[:5])

            rets.append(dict)

        path = os.path.join(argv[4], '{}_{}_{}_{}.{}'.format('new', filename, 'seg2train', 'subsegin3test', 'csv'))
        saverettocsvlr(rets, path)



if __name__ == '__main__':
    main(sys.argv)

