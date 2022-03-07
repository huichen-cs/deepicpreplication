import glob
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

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
    elif 'commitdate' in df.columns:
        df['commitdate'] = pd.to_datetime(df['commitdate'])
        # print(df['commitdate'][:2])
        df.set_axis(df['commitdate'], inplace=True)

        # preprocessing data
        df.drop(columns=['commitdate', 'nm', 'transactionid', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)

    return df

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

def countcomp(df):
    bigger = np.sum(df.iloc[:,1]>df.iloc[:,2])
    smaller = np.sum(df.iloc[:,1]<df.iloc[:,2])
    equal = np.sum(df.iloc[:, 1] == df.iloc[:, 2])
    return bigger, smaller, equal

def plot(df):
    ax = plt.gca()
    df.plot(kind='line', x=df.columns.values[0], y=df.columns.values[1], color='red', ax=ax)
    df.plot(kind='line', x=df.columns.values[0], y=df.columns.values[2], ax=ax)

    plt.show()

def probabilitycal(dfcc, dfcb, dfbc, dfbb, df):
    totalpairs = dfcc.shape[0] + dfcb.shape[0] + dfbc.shape[0] + dfbb.shape[0]
    Pmargin0 = (df[df.bug == 0].shape[0]) / len(df)
    Pmargin1 = (df[df.bug == 1].shape[0]) / len(df)

    Pcondition01 = dfcb.shape[0] / (dfcc.shape[0] + dfcb.shape[0])
    Pcondition11 = dfbb.shape[0] / (dfbc.shape[0] + dfbb.shape[0])
    Pcondition00 = dfcc.shape[0] / (dfcc.shape[0] + dfcb.shape[0])
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

def probabilitycalfromcsv(csvfn):
    print(csvfn)
    # sys.argv[1] filename = "C:\\Users\\ZhaoY\\Downloads\\defectpredict\\download\\kamei2012\\input\\bugzilla.csv"
    df = readsortdata(csvfn)
    
    _, dfcc, dfcb, dfbc, dfbb = getpairs(df)

    ret = probabilitycal(dfcc, dfcb, dfbc, dfbb, df)
    ret['project'] = csvfn.split("\\")[-1][0:-4]
    # if 0 == len(outputdf):
    #     outputdf = ret
    # else:
    #     outputdf = pd.concat([outputdf, ret], axis=0)    
    
    return ret, dfcc, dfcb, dfbc, dfbb

def getmutipairs(df):
    indexccc = []
    entropy1ccc = []
    entropy2ccc = []
    entropy3ccc = []

    indexcbc = []
    entropy1cbc = []
    entropy2cbc = []
    entropy3cbc = []

    indexbcc = []
    entropy1bcc = []
    entropy2bcc = []
    entropy3bcc = []

    indexccb = []
    entropy1ccb = []
    entropy2ccb = []
    entropy3ccb = []

    indexcbb = []
    entropy1cbb = []
    entropy2cbb = []
    entropy3cbb = []

    indexbcb = []
    entropy1bcb= []
    entropy2bcb = []
    entropy3bcb = []

    indexbbc = []
    entropy1bbc = []
    entropy2bbc = []
    entropy3bbc = []

    indexbbb = []
    entropy1bbb = []
    entropy2bbb = []
    entropy3bbb = []

    for ind in range(df.shape[0]-2):
        if (0 == df['bug'][ind]):
            if(0 == df['bug'][ind+1]):
                if (0 == df['bug'][ind + 2]):
                    indexccc.append(ind)
                    entropy1ccc.append(df['entropy'][ind])
                    entropy2ccc.append(df['entropy'][ind+1])
                    entropy3ccc.append(df['entropy'][ind + 2])
                else:
                    indexccb.append(ind)
                    entropy1ccb.append(df['entropy'][ind])
                    entropy2ccb.append(df['entropy'][ind + 1])
                    entropy3ccb.append(df['entropy'][ind + 2])
            else:
                if (0 == df['bug'][ind + 1]):
                    indexcbc.append(ind)
                    entropy1cbc.append(df['entropy'][ind])
                    entropy2cbc.append(df['entropy'][ind + 1])
                    entropy3cbc.append(df['entropy'][ind + 2])
                else:
                    indexcbb.append(ind)
                    entropy1cbb.append(df['entropy'][ind])
                    entropy2cbb.append(df['entropy'][ind + 1])
                    entropy3cbb.append(df['entropy'][ind + 2])
        else:
            if (0 == df['bug'][ind + 1]):
                if (0 == df['bug'][ind + 2]):
                    indexbcc.append(ind)
                    entropy1bcc.append(df['entropy'][ind])
                    entropy2bcc.append(df['entropy'][ind + 1])
                    entropy3bcc.append(df['entropy'][ind + 2])
                else:
                    indexbcb.append(ind)
                    entropy1bcb.append(df['entropy'][ind])
                    entropy2bcb.append(df['entropy'][ind + 1])
                    entropy3bcb.append(df['entropy'][ind + 2])
            else:
                if (0 == df['bug'][ind + 1]):
                    indexbbc.append(ind)
                    entropy1bbc.append(df['entropy'][ind])
                    entropy2bbc.append(df['entropy'][ind + 1])
                    entropy3bbc.append(df['entropy'][ind + 2])
                else:
                        indexbbb.append(ind)
                        entropy1bbb.append(df['entropy'][ind])
                        entropy2bbb.append(df['entropy'][ind + 1])
                        entropy3bbb.append(df['entropy'][ind + 2])

    cctuples = list(zip(indexccc, entropy1ccc, entropy2ccc, entropy3ccc))
    dfccc = pd.DataFrame(cctuples, columns=['cccindex','1','2', '3'])

    cctuples = list(zip(indexccb, entropy1ccb, entropy2ccb, entropy3ccb))
    dfccb = pd.DataFrame(cctuples, columns=['ccbindex', '1', '2', '3'])

    cctuples = list(zip(indexcbc, entropy1cbc, entropy2cbc, entropy3cbc))
    dfcbc = pd.DataFrame(cctuples, columns=['cbcindex', '1', '2', '3'])

    cctuples = list(zip(indexbcc, entropy1bcc, entropy2bcc, entropy3bcc))
    dfbcc = pd.DataFrame(cctuples, columns=['bccindex', '1', '2', '3'])

    cctuples = list(zip(indexcbb, entropy1cbb, entropy2cbb, entropy3cbb))
    dfcbb = pd.DataFrame(cctuples, columns=['cbbindex', '1', '2', '3'])

    cctuples = list(zip(indexbcb, entropy1bcb, entropy2bcb, entropy3bcb))
    dfbcb = pd.DataFrame(cctuples, columns=['bcbindex', '1', '2', '3'])

    cctuples = list(zip(indexbbc, entropy1bbc, entropy2bbc, entropy3bbc))
    dfbbc = pd.DataFrame(cctuples, columns=['bbcindex', '1', '2', '3'])

    cctuples = list(zip(indexbbb, entropy1bbb, entropy2bbb, entropy3bbb))
    dfbbb = pd.DataFrame(cctuples, columns=['bbbindex', '1', '2', '3'])

    return dfccc, dfccb, dfcbc, dfbcc, dfcbb, dfbcb, dfbbc, dfbbb


