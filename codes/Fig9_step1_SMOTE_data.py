# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import pandas as pd
import os
import sys
import glob
import collections
from onlinelearning.RQ3.rq3publicfuncs import saveclndefectdftocsv, conceptdrftsplt, readsortdatawithcommitdate, \
                        doubleclean, savesmotedataapache, getsmotegerateddata, preprocessing

# "C:\Users\ZhaoY\Downloads\results\dropbox\Fig3_results\segments"
# "C:\Users\ZhaoY\Downloads\results\dropbox\FIg11\SMOTEnoPCbalaceddata"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        # files = "C://Users//ZhaoY//Downloads//results//question3//segments//bug_segment_0.csv"
        filename = os.path.basename(files)[:3]
        df = pd.read_csv(files)

        counter = collections.Counter(df['bug'])
        print(counter)
        # spatialnature(counter, df.loc[:, df.columns != 'bug'].to_numpy(), df['bug'], 2, 3)

        # balancedpath = "C://Users//ZhaoY//Downloads//results//question3//resultswithoutSMOTEPC//bug_segment_0_noPC.csv"
        balancedpath = os.path.join(argv[2], '{}_{}_{}.{}'.format(filename, "directSMOTEnoPC", os.path.basename(files)[-5], 'csv'))
        balanceddf = savesmotedataapache(df, balancedpath)
        counterbal = collections.Counter(balanceddf['bug'])
        print(counterbal)


if __name__ == '__main__':
    main(sys.argv)

