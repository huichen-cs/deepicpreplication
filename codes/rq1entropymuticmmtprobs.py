import glob
import os
import sys
import pandas as pd
from onlinelearning.RQ1.rq1publicfuncs import readsortdata, getmutipairs
def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    # "C:\\Users\\ZhaoY\\Downloads\\defectpredict\\download\\kamei2012\\input\\bugzilla.csv"
    datapath = os.path.join(argv[1], '{}_{}.{}'.format("*","pdszz", 'csv'))
    filenames = glob.glob(datapath)

    output = pd.DataFrame()

    for files in filenames:
        print(files)
        # sys.argv[1] filename = "C:\\Users\\ZhaoY\\Downloads\\defectpredict\\download\\kamei2012\\input\\bugzilla.csv"
        df = readsortdata(files)

        dfccc, dfccb, dfcbc, dfbcc, dfcbb, dfbcb, dfbbc, dfbbb = getmutipairs(df)
        totalpairs = dfccc.shape[0] + dfccb.shape[0] + dfcbc.shape[0] + dfbcc.shape[0]  \
                    + dfbbc.shape[0] + dfbcb.shape[0] + dfcbb.shape[0] + dfbbb.shape[0]
        ret = {"ccc": [dfccc.shape[0]/totalpairs], "ccb": [dfccb.shape[0] / totalpairs], "cbc": [dfcbc.shape[0] / totalpairs],
               "bcc": [dfbcc.shape[0] / totalpairs],
               "cbb": [dfcbb.shape[0] / totalpairs], "bcb": [dfbcb.shape[0] / totalpairs],
               "bbc": [dfbbc.shape[0] / totalpairs], "bbb": [dfbbb.shape[0] / totalpairs]}
        ret = pd.DataFrame.from_dict(ret)
        if 0 == len(output):
            output = ret
        else:
            output = pd.concat([output, ret], axis=0)

    outputpath = os.path.join(argv[2], '{}.{}'.format("rq1_multipairprobabilities", 'csv'))
    output.to_csv(outputpath, index=False)




if __name__ == '__main__':
    print(sys.argv)
    main(sys.argv)