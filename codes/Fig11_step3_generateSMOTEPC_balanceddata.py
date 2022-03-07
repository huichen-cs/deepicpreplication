# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys

import pandas as pd
import collections
from onlinelearning.RQ3.rq3publicfuncs import generatebalanceddataset

# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\segments"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\smotegeneratedminoritybeforeR"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\index"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\fulldatafromindex"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\resultswithSMOTEPCfromR\balanceddata"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)
    print(argv[1], argv[2], argv[3], argv[4], datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        print(filename)
        segnum = os.path.basename(files)[-5]
        df = pd.read_csv(files)
        print(df.head())

        counter = collections.Counter(df['bug'])
        num = counter[0] - counter[1]
        print(counter[0], counter[1], num)

        smotegeneratedminoritypath = os.path.join(argv[2], '{}_{}_{}.{}'.format(filename, "smgeneratedmino", segnum, 'csv'))

        # this part needs to be ran after R code, since here we need the distances to principal curve
        Rselecetedindexpath = os.path.join(argv[3], '{}_{}_{}.{}'.format(filename, "seg", segnum, 'csv'))
        selectedsmotedatabyindexsavepath = os.path.join(argv[4], '{}_{}_{}.{}'.format(filename, "smseldata", segnum, 'csv'))
        combineselectedsmotedatawithrealspath = os.path.join(argv[5],
                                                             '{}_{}_{}.{}'.format(filename, "combselsmdataori", segnum, 'csv'))
        print(smotegeneratedminoritypath, Rselecetedindexpath,selectedsmotedatabyindexsavepath, combineselectedsmotedatawithrealspath)

        ret = generatebalanceddataset(df, num,
                                      smotegeneratedminoritypath, Rselecetedindexpath,
                                      selectedsmotedatabyindexsavepath,
                                      combineselectedsmotedatawithrealspath)

        df = pd.read_csv(combineselectedsmotedatawithrealspath)
        counter = collections.Counter(df['bug'])
        print(counter)




if __name__ == '__main__':
    main(sys.argv)

