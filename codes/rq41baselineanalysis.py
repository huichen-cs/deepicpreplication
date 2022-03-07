import math
import os
import sys
import glob
import pandas as pd

# "C:\Users\ZhaoY\Downloads\results\question4\4.1\baseline"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)

    for files in filenames:
        filename = os.path.basename(files)[6:-4]
        if filename not in ["openjpa_pdszz", "camel_pdszz", "hadoop_pdszz", "hbase_pdszz"]:
            continue
        df = pd.read_csv(files)

        auc_means = df['auc'].mean()
        f1_means = df['f1'].mean()

        auc_sum = 0
        f1_sum = 0
        for index, row in df.iterrows():
            auc_sum = auc_sum+math.sqrt((row['auc']-auc_means)**2)
            f1_sum = f1_sum + math.sqrt((row['f1'] - auc_means) ** 2)

        print(filename)
        print(auc_means, auc_sum/len(df))
        print(f1_means, f1_sum / len(df))




if __name__ == '__main__':
    main(sys.argv)











