import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from itertools import chain
from sklearn.preprocessing import StandardScaler

def plotdist(df):
    # check bug distribution by month
    df_by_month = df.resample('M').sum()
    sns.lineplot(x=df_by_month.index, y='bug',data=df_by_month)
    plt.show()


def getmonthindex(df, window):
    column_indices  = {name: i for i, name in enumerate(df.columns)}
    monthindex = column_indices['month']
    print(column_indices)
    print(monthindex)
    tempmonth = df['month'][0]
    cnt = 1
    indexlist = []
    templist = []
    for index in range(len(df)):
        if tempmonth != df.iloc[index, monthindex]:
            cnt = cnt + 1
            tempmonth = df.iloc[index, monthindex]

        if cnt < window+1:
            templist.append(index)
        else:
            indexlist.append(templist)
            # print(len(templist))
            templist = []
            cnt = 1
            templist.append(index)

    if len(templist)>0:
        indexlist.append(templist)
    return indexlist

def getslidingmonthindex(df, window, sliding):
    tempmonth = df['month'][0]
    indexlist = []
    templist = []

    for index, row in df.iterrows():
        if tempmonth != row['month']:
            tempmonth = row['month']
            indexlist.append(templist)
            templist = []
        else:
            templist.append(index)
    ret = []
    i = 0
    while(i<len(indexlist)-window):
        ret.append(list(chain.from_iterable(indexlist[i:(i+window)])))
        i = i+window-sliding

    return ret
def readsortdatanodrop(filename):
    # read data
    df = pd.read_csv(filename)
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        # print(df['commitdate'][:2])
        df.set_axis(df['date_time'], inplace=True)

        # preprocessing data
        df.drop(columns=['date_time', 'time_stamp', 'commit_id', 'nd', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='date_time', axis=0, ascending=True, inplace=True)
    elif 'commitdate' in df.columns:
        df['commitdate'] = pd.to_datetime(df['commitdate'])
        # print(df['commitdate'][:2])
        df.set_axis(df['commitdate'], inplace=True)

        # preprocessing data
        df.drop(columns=['commitdate', 'transactionid', 'nm', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)

    return df

def readsortdata(filename):
    # read data
    df = pd.read_csv(filename)
    if 'date_time' in df.columns:
        df['date_time'] = pd.to_datetime(df['date_time'])
        # print(df['commitdate'][:2])
        df.set_axis(df['date_time'], inplace=True)

        # preprocessing data
        df.drop(columns=['date_time', 'time_stamp', 'commit_id'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='date_time', axis=0, ascending=True, inplace=True)
    elif 'commitdate' in df.columns:
        df['commitdate'] = pd.to_datetime(df['commitdate'])
        # print(df['commitdate'][:2])
        df.set_axis(df['commitdate'], inplace=True)

        # preprocessing data
        df.drop(columns=['commitdate', 'transactionid'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)

    return df

def preprocessing_log(df):
    logcolumns = df.columns
    df.loc[:, logcolumns] = np.log(df[logcolumns]+1)

    return df

def preprocessing(df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df.values)
    df = pd.DataFrame(scaled_features, index=df.index, columns=df.columns)

    return df

def probabilitycal(dfcc, dfcb, dfbc, dfbb, df):
    totalpairs = dfcc.shape[0] + dfcb.shape[0] + dfbc.shape[0] + dfbb.shape[0]
    Pmargin0 = (df[df.bug == 0].shape[0]) / len(df)
    Pmargin1 = (df[df.bug == 1].shape[0]) / len(df)

    if 0 == (dfcc.shape[0] + dfcb.shape[0]):
        Pcondition01 = 0
    else:
        Pcondition01 = dfcb.shape[0] / (dfcc.shape[0] + dfcb.shape[0])

    if 0 == (dfbc.shape[0] + dfbb.shape[0]):
        Pcondition11 = 0
    else:
        Pcondition11 = dfbb.shape[0] / (dfbc.shape[0] + dfbb.shape[0])

    if 0 == (dfcc.shape[0] + dfcb.shape[0]):
        Pcondition00 = 0
    else:
        Pcondition00 = dfcc.shape[0] / (dfcc.shape[0] + dfcb.shape[0])

    if 0 == (dfbc.shape[0] + dfbb.shape[0]):
        Pcondition10 = 0
    else:
        Pcondition10 = dfbc.shape[0] / (dfbc.shape[0] + dfbb.shape[0])

    Ppair00 = dfcc.shape[0] / totalpairs
    Ppair01 = dfcb.shape[0] / totalpairs
    Ppair10 = dfbc.shape[0] / totalpairs
    Ppair11 = dfbb.shape[0] / totalpairs
    Pjoint00 = Pmargin0 * Pcondition00
    Pjoint01 = Pmargin0 * Pcondition01
    Pjoint10 = Pmargin1 * Pcondition10
    Pjoint11 = Pmargin1 * Pcondition11
    # print(Pmargin1, Pcondition01, Pcondition11, Pmargin0,Pcondition00, Pcondition10)
    ret = {"Pcondition01": [Pcondition01], "Pcondition00": [Pcondition00], "Pcondition10": [Pcondition10], "Pcondition11": [Pcondition11],
           "Pmargin0": [Pmargin0], "Pmargin1": [Pmargin1],
           "Ppair00": [Ppair00], "Ppair01": [Ppair01], "Ppair10": [Ppair10], "Ppair11": [Ppair11],
           "Pjoint00": [Pjoint00], "Pjoint01": [Pjoint01], "Pjoint10": [Pjoint10], "Pjoint11": [Pjoint11]}
    ret = pd.DataFrame.from_dict(ret)
    return ret

def getpairs(df):
    indexcc = []
    entropy1cc = []
    entropy2cc = []

    indexcb = []
    entropy1cb = []
    entropy2cb = []

    indexbb = []
    entropy1bb = []
    entropy2bb = []

    indexbc = []
    entropy1bc = []
    entropy2bc = []

    for ind in range(df.shape[0]-1):
        if (0 == df['bug'][ind]):
            if(0 == df['bug'][ind+1]):
                indexcc.append(ind)
                entropy1cc.append(df['entropy'][ind])
                entropy2cc.append(df['entropy'][ind+1])
            else:
                indexcb.append(ind)
                entropy1cb.append(df['entropy'][ind])
                entropy2cb.append(df['entropy'][ind + 1])
        else:
            if (0 == df['bug'][ind + 1]):
                indexbc.append(ind)
                entropy1bc.append(df['entropy'][ind])
                entropy2bc.append(df['entropy'][ind + 1])
            else:
                indexbb.append(ind)
                entropy1bb.append(df['entropy'][ind])
                entropy2bb.append(df['entropy'][ind + 1])

    ret = {"cleanbugindex": indexcb, "cleanbugentropy1":entropy1cb, "cleanbugentropy2":entropy2cb,
           "cleancleanindex": indexcc, "cleancleanentropy1":entropy1cc, "cleancleanentropy2":entropy2cc,
           "bugbugindex": indexbb, "bugbugentropy1":entropy1bb, "bugbugentropy2":entropy2bb,
           "bugcleanindex": indexbc, "bugcleanentropy1":entropy1bc, "bugcleanentropy2":entropy2bc}

    cctuples = list(zip(indexcc, entropy1cc, entropy2cc))
    dfcc = pd.DataFrame(cctuples, columns=['cleancleanindex','cleancleanentropy1','cleancleanentropy2'])

    cctuples = list(zip(indexcb, entropy1cb, entropy2cb))
    dfcb = pd.DataFrame(cctuples, columns=['cleanbugindex', 'cleanbugentropy1', 'cleanbugentropy2'])

    cctuples = list(zip(indexbc, entropy1bc, entropy2bc))
    dfbc = pd.DataFrame(cctuples, columns=['bugcleanindex', 'bugcleanentropy1', 'bugcleanentropy2'])

    cctuples = list(zip(indexbb, entropy1bb, entropy2bb))
    dfbb = pd.DataFrame(cctuples, columns=['bugbugindex', 'bugbugentropy1', 'bugbugentropy2'])

    return ret, dfcc, dfcb, dfbc, dfbb

def conceptdrftsplt(df):
    n = df.shape[0]
    df1 = df[0:int(n / 5)]
    df2 = df[int(n / 5):int(n / 5) * 2]
    df3 = df[int(n / 5) * 2:int(n / 5) * 3]
    df4 = df[int(n / 5) * 3:int(n / 5) * 4]
    df5 = df[int(n / 5) * 4:]
    return [df1, df2, df3, df4, df5]