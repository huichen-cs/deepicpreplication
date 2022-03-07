import atexit
import dropbox
import io
import logging
import numpy as np
import os
import pandas as pd
import sys
import posixpath
from matplotlib import pyplot as plt
from scipy.spatial import distance
from dropboxtool.dropboxutils import DropboxUtils, load_app_key




logger = logging.getLogger(__name__)

__dbx_key_dir = '../../../secret'
__dbx_token_dir = '../../../secret'
__dbx_root_dir = 'onlinelearningrets'
__dbx = None

def set_dbx_parameters(key_dir, token_dir, root_dir):
    global __dbx_key_dir, __dbx_token_dir, __dbx_root_dir
    __dbx_key_dir, __dbx_token_dir, __dbx_root_dir = key_dir, token_dir, root_dir


def open_dropbox():
    global __dbx
    app_key = load_app_key(key_file=None, key_dir=__dbx_key_dir)
    dropbox_utils = DropboxUtils(__dbx_token_dir, app_key)
    refresh_token = dropbox_utils.get_refresh_token()
    __dbx =  dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=app_key)


def close_dropbox():
    global __dbx
    logging.info('closing dropbox ......')
    if not __dbx is None:
        __dbx.close()

def read_dropbox_data(filename):
    global __dbx
    # app_key = load_app_key(key_file=None, key_dir=__dbx_key_dir)
    # dropbox_utils = DropboxUtils(__dbx_token_dir, app_key)
    # refresh_token = dropbox_utils.get_refresh_token()
    # with dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=app_key) as dbx:
    file_from = posixpath.join('/', __dbx_root_dir, filename)
    _, res = __dbx.files_download(file_from, rev=None)
        # print(type(res))
    return pd.read_csv(io.StringIO(str(res.content, 'utf-8')))

def read_data(fn1, source_type='dropbox'):
    if source_type == 'dropbox':
        return read_dropbox_data(fn1)
    else:
        raise ValueError('unsupported data source type {}'.format(source_type))


def normalize_df(df):
    return (df-df.min())/(df.max()-df.min())

def cmp_length(df):
    return (df.iloc[0:-1].reset_index(drop=True) - df.iloc[1:].reset_index(drop=True))\
        .pow(2).sum(axis=1).pow(0.5)

def interpolate_df(df, npts=101):
    segs_len = cmp_length(df)
    segs_len_cumsum = np.zeros(len(segs_len)+1)
    segs_len_cumsum[1:] = np.cumsum(segs_len)
    tot_len = segs_len.sum()
    step_len = tot_len/(npts - 1)
    segs_len_new = np.linspace(0, npts-1, npts) * step_len
    row_list = []
    for _,y in enumerate(segs_len_new):
        found = False
        for i,s in enumerate(segs_len_cumsum):
            if y <= s:
                if i == 0:
                    found = True
                    row_list.append(df.iloc[i])
                else:
                    found = True
                    x1 = df.iloc[i-1]
                    x2 = df.iloc[i]
                    np.testing.assert_almost_equal(np.sqrt((x2 - x1).pow(2).sum()), segs_len[i-1])
                    row_list.append((x2 - x1)*(s-y)/segs_len[i-1] + x1)
                break
        if not found:
            row_list.append(df.iloc[-1])
            # else:
            #     print(j, i, y, s)
    # print(len(row_list))
    return pd.DataFrame(row_list)

def interpolate_pcs(df_pc_list, npts=101):
    pc_list = []
    for pc in df_pc_list:
        # normalize_df(pc)
        pc_list.append(interpolate_df(normalize_df(pc), npts=npts))
    return pc_list

def cmp_pc_similarity(df1n, df2n, dist_func, columns=None, npts=101):
    # df1, df2 = normalize_df(df1), normalize_df(df2)
    # df1n = interpolate_df(df1, npts=npts)
    # df2n = interpolate_df(df2, npts=npts)
    
    
    # from matplotlib import pyplot as plt
    # plt.plot(df1n['la'], df1n['nf'], color='black', linestyle='dashed')
    # plt.plot(df1['la'], df1['nf'], color='red', linestyle='dotted')
    
    # from sklearn.metrics.pairwise import cosine_similarity
    # print(np.diag(cosine_similarity(df1n, df1n)).mean())
    # print(np.diag(cosine_similarity(df1n, df2n)).mean())

    # print(1-distance_function(df1n.to_numpy().flatten(), df2n.to_numpy().flatten())) 
    
    # plt.plot(df2n['la'], df2n['nf'], color='black', linestyle='dashed')
    # plt.plot(df2['la'], df2['nf'], color='red', linestyle='dotted')
    # plt.show() 
    logging.debug('df1n.columns = {}'.format(df1n.columns))
    logging.debug('df2n.columns = {}'.format(df2n.columns))

    return 1 - dist_func(df1n.to_numpy().flatten(), df2n.to_numpy().flatten())


def cmp_pc_similarity_from_files(d_func, files, columns=None):
    pc_list = []
    for f in files:
        pc_list.append(pd.read_csv(f))
    d_list = pc_list
    
    for i in range(len(d_list)):
        df = d_list[i]
        if not columns is None \
            and set(df.columns[df.columns.isin(columns)]) == set(columns):
            df = df[columns]
        elif not columns is None:
            raise ValueError('columns {} not all present in the data, df.colulmns={}'
                             .format(columns, df.columns))
        logging.debug('df.columns = {}'.format(df.columns))
        d_list[i] = df
    
    d_mtx = np.zeros((len(d_list), len(d_list)))
    for i,d1 in enumerate(d_list):
        for j,d2 in enumerate(d_list):
            d_mtx[i, j] = cmp_pc_similarity(d1, d2, d_func, columns)
    return d_mtx


def get_projects(folder):
    global __dbx
    # app_key = load_app_key(key_file=None, key_dir=__dbx_key_dir)
    # dropbox_utils = DropboxUtils(__dbx_token_dir, app_key)
    # refresh_token = dropbox_utils.get_refresh_token()
    # with dropbox.Dropbox(oauth2_refresh_token=refresh_token, app_key=app_key) as dbx:
    folder_from = posixpath.join('/', __dbx_root_dir, folder)        
    response = __dbx.files_list_folder(folder_from)
    projects = set()
    for fm in response.entries:
        projects.add(posixpath.split(fm.path_display)[1].split('_')[0])
    return projects

def get_projects_bug_ratios(file_from, source_type='dropbox'):
    return read_data(file_from, source_type=source_type)

def heatmap(mtx, segments, labels, cmap='gray', vmin=-1, vmax=1, 
            lbl_fontsize=16, tick_fontsize=14, text_fontsize=14, 
            text_fmt='{:5.2f}', 
            fig_ax = (None, None), show=False, figfn=None):
    if fig_ax[0] is None or fig_ax[1] is None: 
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    ax.imshow(mtx, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    
    # Show all ticks and label them with the respective list entries
    # segments = [1, 2, 3, 4, 5]
    # labels = ['1', '2', '3', '4', '5']
    # segments = [3, 4]
    # labels = ['3', '4']
    ax.xaxis.tick_top() # x axis on top
    ax.xaxis.set_label_position('top')
    ax.set_xticks(np.arange(len(segments)))
    ax.set_xticklabels(labels=labels, fontsize=tick_fontsize)
    ax.set_yticks(np.arange(len(segments)))
    ax.set_yticklabels(labels=labels, fontsize=tick_fontsize)
    ax.set_xlabel('Segment #', fontsize=lbl_fontsize)
    ax.set_ylabel('Segment #', fontsize=lbl_fontsize)
    
    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    for i in range(len(segments)):
        for j in range(len(segments)):
            if i != j:
                print(mtx[i, j], text_fmt.format(mtx[i, j]))
            if i != j and text_fmt.format(mtx[i, j]) == ' 1.00':
                print('----------------')
                text_fmt = '{:6.3f}'
    logger.info('text_fmt = {}'.format(text_fmt))
    for i in range(len(segments)):
        for j in range(len(segments)):
            ax.text(j, i, text_fmt.format(mtx[i, j]),
                           ha="center", va="center", color="k", fontsize=text_fontsize)
    
    # ax.set_title("Harvest of local farmers (in tons/year)")
    fig.tight_layout()
    if not figfn is None:
        plt.savefig(figfn, bbox_inches="tight")
    if show:
        plt.show()

def get_segment_file_list(project, nsegments):
    return ['{}_segment_{}'.format(project, i) for i in range(nsegments)]

def plot_project_bugratios(project, df_bug_ratios, fig_ax = (None, None), factor=1, show=False):
    if fig_ax[0] is None or fig_ax[1] is None: 
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax
    segment_list = get_segment_file_list(project, 5)
    df = df_bug_ratios.loc[df_bug_ratios['segment'].isin(segment_list)]
    df = df.sort_values(by=['segment'], axis=0, ascending=True)

    width = 0.35
    x = range(1, df.shape[0]+1)
    ax.bar(x, df['clean/buggy'], width, label='Clean/Defect Ratio',
            color='gray')
    ax.set_xlabel('Segment Number', fontsize=16*0.25/0.20*factor)
    ax.set_ylabel('C/D', fontsize=16*0.25/0.20*factor)
    # ax.set_xticklabels(x, fontsize=14*0.25/0.20*factor)
    # ax.set_yticks(fontsize=14*0.25/0.20*factor)
    ax.set_title(project)
    fig.tight_layout()
    # plt.savefig(sys.argv[2], bbox_inches="tight")
    if show:
        plt.show()
 

def main(argv):
    atexit.register(close_dropbox)
    logging.info('working from cwd = {}'.format(os.getcwd()))
    # if len(argv) < 3:
    #     print('Usage: {} pc_1_file.csv pc_2_file_csv'.format(argv[0]))
    #     pc1_file = 'onlinelearning/RQ2/Fig4_pc_data/bug_segment_0.csv'
    #     pc2_file = 'onlinelearning/RQ2/Fig4_pc_data/bug_segment_1.csv'
    #     pc3_file = 'onlinelearning/RQ2/Fig4_pc_data/bug_segment_2.csv'
    #     pc4_file = 'onlinelearning/RQ2/Fig4_pc_data/bug_segment_3.csv'
    #     pc5_file = 'onlinelearning/RQ2/Fig4_pc_data/bug_segment_4.csv'
    #     pc4_file = 'onlinelearning/RQ2/Fig4_pc_data/pos_segment_3.csv'
    #     pc5_file = 'onlinelearning/RQ2/Fig4_pc_data/pos_segment_4.csv'
    # else:
    #     pc1_file, pc2_file = argv[1:3]

    open_dropbox()
    
    projects_bug_ratios = get_projects_bug_ratios('onlinelearning/RQ2/Fig3_results/segments_bugratio/bugratio_segment.csv')
    
    projects = get_projects('onlinelearning/RQ2/Fig4_pc_data/')
    for p in projects:
        file_list = ['onlinelearning/RQ2/Fig4_pc_data/{}_segment_{}.csv'.format(p, i) for i in range(5)]
        

        # d_mtx = cmp_pc_similarity_from_files(distance.correlation, [pc1_file, pc2_file, pc3_file, pc4_file, pc5_file])
        d_mtx = cmp_pc_similarity_from_files(distance.cosine, file_list, ['la', 'nf'])
        
        logger.info('project {}: \n{}'.format(p, d_mtx))
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 6), gridspec_kw={'height_ratios': [1, 2]})
        plot_project_bugratios(p, projects_bug_ratios, fig_ax=(fig, axes[0]))
        heatmap(d_mtx, [1, 2, 3, 4, 5], ['1', '2', '3', '4', '5'], fig_ax=(fig, axes[1]), show=True)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv)
