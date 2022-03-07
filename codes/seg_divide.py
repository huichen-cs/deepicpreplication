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
# original segment dividing
#"C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\segments"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio4"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio8"

# SMOTEPC segment dividing
#"C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\balanceddata"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio4SMOTEPC"
# "C:\Users\ZhaoY\Downloads\results\dropbox\divide_segment\ratio8MOTEPC"
def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        segnum = int(os.path.basename(files)[-5])
        if (segnum!=2):
            continue
        print(segnum)

        df = pd.read_csv(files)
        df_len = len(df)
        print(filename, segnum, df_len)
        for i in range(8):
            print(i)
            if i<4:
                sub_seg_ratio4 = df.loc[df_len*i/4:df_len*(i+1)/4, :]
                path4 = os.path.join(argv[2], '{}_{}_{}_{}.{}'.format(filename, 'seg3', 'subseg', i, 'csv'))
                sub_seg_ratio4.to_csv(path4)

            sub_seg_ratio8 = df.loc[df_len*i/8:df_len*(i+1)/8, :]
            path8 = os.path.join(argv[3], '{}_{}_{}_{}.{}'.format(filename, 'seg3', 'subseg', i, 'csv'))
            sub_seg_ratio8.to_csv(path8)



if __name__ == '__main__':
    main(sys.argv)

