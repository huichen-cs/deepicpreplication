# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys
import time
import datetime
import pandas as pd
from onlinelearning.RQ3.rq3publicfuncs import saveclndefectdftocsv, conceptdrftsplt, readsortdatawithcommitdate, \
                        doubleclean, savesmotedata, getsmotegerateddata, preprocessing, readsortdata

# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\segments"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\segments\clean"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\segments\defect"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\SMOTEPCbeforeselectbalacedata"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\smotegeneratedminoritybeforeR"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)
    #print(argv[1], argv[2], datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        segnum = os.path.basename(files)[-5]

        df = pd.read_csv(files)
        clnpath = os.path.join(argv[2], '{}_{}_{}.{}'.format(filename, "clean", segnum, 'csv'))
        dfectpath = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, "defect", segnum, 'csv'))
        ratio = saveclndefectdftocsv(df, clnpath, dfectpath)
        print(ratio)

        # generate smote data based on real clean num * 2, then future rank disctance first then select shortest ones
        doubledefect_train_df = doubleclean(df)

        smotebeforeselectpath = os.path.join(argv[4], '{}_{}_{}.{}'.format(filename, "smbeforeselect", segnum,'csv'))
        balanceddf = savesmotedata(doubledefect_train_df, smotebeforeselectpath)

        # real smote() generated data which is seperated with original data
        smotegeneratedminoritypath = os.path.join(argv[5], '{}_{}_{}.{}'.format(filename, "smgeneratedmino", segnum, 'csv'))
        resdf = getsmotegerateddata(doubledefect_train_df, balanceddf, smotegeneratedminoritypath)


if __name__ == '__main__':
    main(sys.argv)

