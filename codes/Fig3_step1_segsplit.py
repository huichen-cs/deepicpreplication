# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import glob
import os
import sys
import time
import datetime
import pandas as pd
from onlinelearning.RQ2.rq2publicfuncs import conceptdrftsplt, probabilitycal, preprocessing_log,readsortdata,getpairs

#"C:\Users\ZhaoY\Downloads\ApacheProjects\datasets"
# "C:\Users\ZhaoY\Downloads\results\dropbox\Fig3_results\segments"
# "C:\Users\ZhaoY\Downloads\results\dropbox\Fig3_results\segment_condition_prob"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)
    print(argv[1], argv[2], datapath)

    for files in filenames:
        filename = os.path.basename(files)[:3]
        # tempdf = readsortdatawithcommitdate(files)
        tempdf = readsortdata(files)
        tempdf['commitdate'] = tempdf.index.map(str)
        #print(tempdf.head(1))
        for i in range(tempdf.shape[0]):
            tempdf.commitdate[i] = time.mktime(datetime.datetime.strptime(tempdf.commitdate[i], "%Y-%m-%d %H:%M:%S").timetuple())

        df = preprocessing_log(tempdf.loc[:, ~tempdf.columns.isin(['bug', 'commitdate'])])
        df['commitdate'] = tempdf['commitdate']
        df['bug'] = tempdf['bug']

        dflst = conceptdrftsplt(df)

        # counter = collections.Counter(df['bug'])
        # spatialnature(counter, df.loc[:, df.columns != 'bug'].to_numpy(), df['bug'], 2, 3)
        tempfilename = ""
        outputdata = pd.DataFrame()
        for i in range(len(dflst)):
            segpath = os.path.join(argv[2], '{}_{}_{}.{}'.format(filename, "segment", i, 'csv'))
            dflst[i].to_csv(segpath, index=False)
            P01 = []
            P11 = []
            P00 = []
            P10 = []
            P1 = []
            # print(dflst[i]['bug'].iloc[0])
            pairs, dfcc, dfcb, dfbc, dfbb = getpairs(dflst[i])
            ret = probabilitycal(dfcc, dfcb, dfbc, dfbb, dflst[i])

            P01.append(ret['Pcondition01'][0])
            P11.append(ret['Pcondition11'][0])
            P00.append(ret['Pcondition00'][0])
            P10.append(ret['Pcondition10'][0])
            P1.append(ret['Pmargin1'][0])

            tuplelist = list(zip(P01, P11, P1, P00, P10))
            entropydata = pd.DataFrame(tuplelist, columns=['P(1|0}', 'P(1|1}', 'P1', 'P(0|0}', 'P(0|1}'])
            # entropydata['dataset'] = os.path.basename(files)[:-4]

            if len(tempfilename) == 0:
                tempfilename = os.path.basename(files)[:-4]
                outputdata = pd.concat([outputdata, entropydata])
            elif os.path.basename(files)[:-4] == tempfilename:
                outputdata = pd.concat([outputdata, entropydata])

        # savepath = os.path.join(argv[3], '{}_{}.{}'.format("seg", os.path.basename(files)[:-4], 'csv'))
        # outputdata.to_csv(savepath, index=False)


if __name__ == '__main__':
    main(sys.argv)

