import random
import os
import numpy as np
import sys
import glob
from sklearn import metrics
from rq41publicfuncs import  WGstruct, readsortdata, datasplit2seg, getreallabels,\
    preprocessingdata, confusionmetrics, cmsave, classify

# "C:\Users\ZhaoY\Downloads\ApacheProjects\datasets"
# "C:\Users\ZhaoY\Downloads\results\question4\4.1\baseline"

def main(argv):
    if len(argv) < 2:
        print('Usage: ' + argv[0] + ' work_path_include_project_name')
        sys.exit(1)

    datapath = os.path.join(argv[1], '{}.{}'.format("*", 'csv'))
    filenames = glob.glob(datapath)
    print(argv[1], argv[2], datapath)

    for files in filenames:
        filename = os.path.basename(files)[:-4]
        # filepath = "C:\\Users\\ZhaoY\\Downloads\\defectpredict\\download\\kamei2012\\input\\bugzilla.csv"
        df = readsortdata(files)

        train_df,val_df,test_df = datasplit2seg(df, 0.9)
        train_df, val_df, test_df = preprocessingdata(train_df,val_df,test_df)

        input_widths = 60
        label_widths = 1
        shiftwidth = 0
        batch = 32
        cmlist = []

        for i in range(0, 50):
            cmdict = {"dataset": filename, "index": i}
            w = WGstruct(input_width=input_widths, label_width=label_widths, shift=shiftwidth,
                    sequence_stride=1, batch_size=batch,
                    train_df=train_df, val_df=val_df, test_df=test_df,
                    label_columns=['bug'])

            y_true = getreallabels(w.test)
            randomlist = []
            for i in range(0, np.array(y_true).shape[0]):
                n = random.uniform(0, 1)
                randomlist.append(n)
            # print(randomlist)
            # print(np.array(randomlist).shape)
            # print(y_true.shape)

            cmdict["auc"] = metrics.roc_auc_score(y_true, randomlist)
            y_pred = classify(np.array(randomlist).reshape(-1, 1))
            cmdict["f1"] = metrics.f1_score(y_true, y_pred, average='macro')

            cm = confusionmetrics(np.array(randomlist), w.test)

            cmdict["cm"] = cm
            cmlist.append(cmdict)

        cmsave(cmlist, argv[2], "ave50", filename)

if __name__ == '__main__':
    main(sys.argv)











