import glob
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from onlinelearning.RQ2.rq2publicfuncs import plotdist

# "C:\Users\ZhaoY\Downloads\results\question2\conceptdriftprobability\month3prob"
# "C:\Users\ZhaoY\Downloads\results\question2\conceptdriftprobability\month3plot"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}_{}.{}'.format("seg", '*', 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        df = pd.read_csv(files)
        print(files)
        df.plot(kind="line")
        # plotdist(df)
        #
        savepath = os.path.join(argv[2], '{}_{}.{}'.format("entropycdfigure", os.path.basename(files)[:-4], 'pdf'))
        plt.savefig(savepath)




if __name__ == '__main__':
    main(sys.argv)
    
    