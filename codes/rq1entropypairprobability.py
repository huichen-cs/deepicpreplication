import glob
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from onlinelearning.RQ1.rq1publicfuncs import probabilitycalfromcsv, countcomp, plot

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    # "C:\\Users\\ZhaoY\\Downloads\\defectpredict\\download\\kamei2012\\input\\bugzilla.csv"
    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    output = pd.DataFrame()
    for files in filenames:
        print(files)
        # # sys.argv[1] filename = "C:\\Users\\ZhaoY\\Downloads\\defectpredict\\download\\kamei2012\\input\\bugzilla.csv"
        # df = readsortdata(files)
        #
        # pairs, dfcc, dfcb, dfbc, dfbb = getpairs(df)
        #
        # ret = probabilitycal(dfcc, dfcb, dfbc, dfbb)
        # ret['project'] = files.split("\\")[-1][0:-4]
        # if 0 == len(output):
        #     output = ret
        # else:
        #     output = pd.concat([output, ret], axis=0)

        retdf, dfcc, dfcb, dfbc, dfbb = probabilitycalfromcsv(files)
        output = pd.concat([output, retdf], axis=0)
        print(len(dfcc), len(dfcb), len(dfbc), len(dfbb))



        # ccpath = os.path.join(argv[2], '{}_{}.{}'.format(files.split("\\")[-1][0:-4], "cc", 'csv'))
        # cbpath = os.path.join(argv[2], '{}_{}.{}'.format(files.split("\\")[-1][0:-4], "cb", 'csv'))
        # bcpath = os.path.join(argv[2], '{}_{}.{}'.format(files.split("\\")[-1][0:-4], "bc", 'csv'))
        # bbpath = os.path.join(argv[2], '{}_{}.{}'.format(files.split("\\")[-1][0:-4], "bb", 'csv'))
        #
        # dfcc.to_csv(ccpath, index=False)
        # dfcb.to_csv(cbpath, index=False)
        # dfbc.to_csv(bcpath, index=False)
        # dfbb.to_csv(bbpath, index=False)

        # entropy compare
        biggercc, smallercc, equalcc = countcomp(dfcc)
        print(biggercc, smallercc, equalcc)
        print(biggercc/(biggercc+smallercc+equalcc), smallercc/(biggercc+smallercc+equalcc),
              equalcc/(biggercc+smallercc+equalcc))

        # plot(dfcc)
        # plot(dfcb)
        # plot(dfbc)
        # plot(dfbb)
    outputpath = os.path.join(argv[2], '{}.{}'.format("rq1_pair_probabilities", 'csv'))
    output.to_csv(outputpath, index=False)




if __name__ == '__main__':
    main(sys.argv)