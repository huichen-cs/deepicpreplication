# https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import collections
from matplotlib import pyplot
from sklearn.preprocessing import StandardScaler


def spatialnature(counter, x, y, col1, col2):
    for label, _ in counter.items():
        row_ix = np.where(y == label)[0]
        pyplot.scatter(x[row_ix, col1], x[row_ix, col2], label=str(label))
    pyplot.legend()
    pyplot.show()


def sourtedbydist(Rselecetedindexpath):
    dfselected = pd.read_csv(Rselecetedindexpath)
    # print(dfselected.head())
    dfselected.sort_values(by='distance', axis=0, ascending=True, inplace=True)
    # print(dfselected.head())
    dfselected.drop(columns='distance', inplace=True)
    # print(dfselected.head())

    return dfselected


def selectedsmotedefects(Rselecetedindexpath, smotegenerateddatapath, selectedsmotedatabyindexsavepath):
    dfselected = sourtedbydist(Rselecetedindexpath)

    dfall = pd.read_csv(smotegenerateddatapath)
    retdf = dfall.loc[dfselected.index]
    retdf.to_csv(selectedsmotedatabyindexsavepath, index=False)

    print(retdf.shape)

    return retdf


def saveclndefectdftocsv(df, pathclean, pathdefect):
    cleans = df[df.bug == 0]
    print(cleans.shape)
    cleans.to_csv(pathclean, index=False)

    defects = df[df.bug == 1]
    defects.to_csv(pathdefect, index=False)

    return cleans.shape[0] / defects.shape[0]


def  savesmotedata(df, balancedpath):
    counterres = collections.Counter(df['bug'])
    print(counterres)
    sm = SMOTE()
    X_res, y_res = sm.fit_resample(df.loc[:, df.columns != 'bug'].to_numpy(), df['bug'])
    counterres = collections.Counter(y_res)
    print(counterres)
    print(X_res.shape)
    print(y_res.shape)

    # spatialnature(counterres, X_res, y_res, 2, 3)
    balanceddf = pd.DataFrame(X_res, columns=df.columns[:-1])
    balanceddf['bug'] = y_res
    print(balanceddf.head())
    balanceddf.to_csv(balancedpath, index=False)

    return balanceddf

def  savesmotedataapache(df, balancedpath):
    counterres = collections.Counter(df['bug'])
    print(counterres)
    histcolumns = {name: i for i, name in enumerate(df.columns)}
    print(histcolumns)

    sm = SMOTE()
    X_res, y_res = sm.fit_resample(df.loc[:, df.columns != 'bug'].to_numpy(), df['bug'])
    counterres = collections.Counter(y_res)
    print(counterres)
    print(X_res.shape)
    print(y_res.shape)

    # spatialnature(counterres, X_res, y_res, 2, 3)
    balanceddf = pd.DataFrame(X_res, columns=df.columns[:-1])
    balanceddf['bug'] = y_res
    print(balanceddf.head())
    balanceddf.to_csv(balancedpath, index=False)

    return balanceddf

def getsmotegerateddata(df, balanceddf, generatordatapath):
    resdf = pd.concat([df, balanceddf]).drop_duplicates(keep=False)
    resdf.to_csv(generatordatapath, index=False)
    print(resdf.info())
    return resdf


def combinesmoteandoriginaldata(df, smdf, savepath):
    frames = [df, smdf]
    result = pd.concat(frames)

    result.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)
    result.drop(columns='commitdate', inplace=True)
    result.to_csv(savepath, index=False)

    return result


def conceptdrftsplt(df):
    n = df.shape[0]
    df1 = df[0:int(n / 5)]
    df2 = df[int(n / 5):int(n / 5) * 2]
    df3 = df[int(n / 5) * 2:int(n / 5) * 3]
    df4 = df[int(n / 5) * 3:int(n / 5) * 4]
    df5 = df[int(n / 5) * 4:]
    return [df1, df2, df3, df4, df5]

def readsortdata(filename):
    # read data
    df = pd.read_csv(filename)
    # print(df.shape)
    # print(df.info())

    # index df by commitdate
    # print(type(df['commitdate']))
    # print(df['commitdate'][:2])
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        # print(df['commitdate'][:2])
        df.set_axis(df['date_time'], inplace=True)

        # preprocessing data
        df.drop(columns=['date_time', 'time_stamp', 'commit_id', 'nd', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='date_time', axis=0, ascending=True, inplace=True)

        df['date_time'] = df.index

    elif 'commitdate' in df.columns:
        df['commitdate'] = pd.to_datetime(df['commitdate'])
        # print(df['commitdate'][:2])
        df.set_axis(df['commitdate'], inplace=True)

        # preprocessing data
        df.drop(columns=['commitdate', 'nm', 'transactionid', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)
        df['commitdate'] = df.index

    return df

def readsortdatawithcommitdate(filename):
    # read data
    df = pd.read_csv(filename)
    df.drop(columns=['nm', 'transactionid', 'rexp'], inplace=True)

    # sort data set by commit time
    df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)

    return df

def readsortdatawithcommitdateforrq3(filename):
    # read data
    df = pd.read_csv(filename)
    if 'date_time' in df.columns:
        df.drop(columns=['time_stamp', 'commit_id', 'nd', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='date_time', axis=0, ascending=True, inplace=True)

    elif 'commitdate' in df.columns:
        df.drop(columns=['nm', 'transactionid', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)

    return df

def readsortdatawithcommitdate4apache(filename):
    # read data
    df = pd.read_csv(filename)
    print(df.head(10))

    df.drop(columns=['time_stamp', 'commit_id', 'nd', 'rexp'], inplace=True)
    print(df.head(10))

    # sort data set by commit time
    df.sort_values(by='date_time', axis=0, ascending=True, inplace=True)
    print(df.head(10))

    return df

def doubleclean(df):
    cleans = df[df.bug == 0]

    frames = [df, cleans]
    ret = pd.concat(frames)

    return ret


def generatebalanceddataset(df, num,
                            smotegeneratedminoritypath, Rselecetedindexpath,
                            selectedsmotedatabyindexsavepath,
                            combineselectedsmotedatawithrealspath
                            ):
    selectedsmotewholedf = selectedsmotedefects(Rselecetedindexpath, smotegeneratedminoritypath,
                                                selectedsmotedatabyindexsavepath)

    ret = combinesmoteandoriginaldata(df, selectedsmotewholedf.head(num), combineselectedsmotedatawithrealspath)
    return ret


def saveparamdictocsv(paramsavepath, paramdict):
    params_items = paramdict.items()
    params_list = list(params_items)
    paramdf = pd.DataFrame(params_list)
    paramdf.to_csv(paramsavepath, index=False)


def preprocessing(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    return df

