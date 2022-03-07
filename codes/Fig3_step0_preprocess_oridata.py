# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys
import time
import datetime
import pandas as pd
from onlinelearning.RQ2.rq2publicfuncs import conceptdrftsplt, probabilitycal, preprocessing_log,readsortdata,getpairs

# "C:\Users\ZhaoY\Downloads\ApacheProjects\datasets_ori"
# "C:\Users\ZhaoY\Downloads\ApacheProjects\datasets"
def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)
    #print(argv[1], argv[2], datapath)

    for files in filenames:
        filename = os.path.basename(files)[:-4]
        df = pd.read_csv(files)
        print(df.shape)
        c = 0
        for i in range(df.shape[0]):
            if df['age'][i]<0:
                df = df.drop(labels=i, axis=0)
                c = c+1
        print(df.shape)
        print(filename, c)
        savepath = os.path.join(argv[2], '{}.{}'.format(filename, 'csv'))
        df.to_csv(savepath, index=False)


if __name__ == '__main__':
    main(sys.argv)

