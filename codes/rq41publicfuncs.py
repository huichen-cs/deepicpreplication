import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tensorflow.keras.models import load_model
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import StandardScaler

METRICS = [
      # tf.keras.metrics.TruePositives(name='tp'),
      # tf.keras.metrics.FalsePositives(name='fp'),
      # tf.keras.metrics.TrueNegatives(name='tn'),
      # tf.keras.metrics.FalseNegatives(name='fn'),
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 sequence_stride, batch_size,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.sequence_stride = sequence_stride
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


# w2 = WindowGenerator(input_width=6, label_width=1, shift=1,
#                      label_columns=['bug'])
#
# print(w2)

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        # print(inputs)
        labels = features[:, self.labels_slice, :]
        # print(labels)
        # print(self.column_indices, self.column_indices['bug'])
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride,
            shuffle=False,
            batch_size=self.batch_size, )

        ds = ds.map(self.split_window)
        # print_dataset(ds)

        return ds

        # WindowGenerator.make_dataset = make_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


    def plot(self, model=None, plot_col='bug', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                      c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('bug')
        plt.show()

    def multiplot(self, model=None, plot_col='bug'):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = len(inputs)
        for n in range(5):
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices+n*self.shift, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            # if label_col_index is None:
            #     continue

            plt.scatter(self.label_indices+n*self.shift, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices+n*self.shift, predictions[n, :, label_col_index],
                      marker='X', edgecolors='k', label='Predictions',
                       s=64)


        plt.legend()

        plt.xlabel('bug')
        plt.show()

def readsortdata(filename):
    # read data
    df = pd.read_csv(filename)

    if 'time_stamp' in df.columns:
        df['time_stamp'] = pd.to_datetime(df['time_stamp'])
        # print(df['commitdate'][:2])
        df.set_axis(df['time_stamp'], inplace=True)
        df.drop(columns=['date_time', 'time_stamp', 'commit_id', 'nd', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='time_stamp', axis=0, ascending=True, inplace=True)

    elif 'commitdate' in df.columns:
        df['commitdate'] = pd.to_datetime(df['commitdate'])
        # print(df['commitdate'][:2])
        df.set_axis(df['commitdate'], inplace=True)

        # preprocessing data
        df.drop(columns=['commitdate', 'nm', 'transactionid', 'rexp'], inplace=True)

        # sort data set by commit time
        df.sort_values(by='commitdate', axis=0, ascending=True, inplace=True)

    return df

def datasplit2seg(df, ratio):
    n = df.shape[0]
    train_df = df[0:int(n * ratio)]
    test_df = df[int(n * ratio):]
    val_df = train_df
    return train_df, val_df, test_df

def datasplit(df, ratio):
    n = df.shape[0]
    train_df = df[0:int(n * ratio[0])]
    val_df = df[int(n * ratio[0]):int(n * ratio[1])]
    test_df = df[int(n * ratio[1]):]
    return train_df, val_df, test_df

def getreallabels(generator):
    ret = []
    for hist, curr, targets in generator:
        for bat in range(len(targets)):
            ret.append(targets[bat][0][0])

    return np.array(ret)

def preprocessing(train_df, val_df, test_df):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(train_df.values)
    train_df = pd.DataFrame(scaled_features, index=train_df.index, columns=train_df.columns)

    scaled_features = scaler.fit_transform(val_df.values)
    val_df = pd.DataFrame(scaled_features, index=val_df.index, columns=val_df.columns)

    scaled_features = scaler.fit_transform(test_df.values)
    test_df = pd.DataFrame(scaled_features, index=test_df.index, columns=test_df.columns)

    return train_df, val_df, test_df



    return train_df, val_df, test_df

def preprocessingdata(temp_train_df, temp_val_df, temp_test_df):
    train_df, val_df, test_df = preprocessing(temp_train_df.loc[:, temp_train_df.columns != 'bug'],
                                              temp_val_df.loc[:, temp_val_df.columns != 'bug'],
                                              temp_test_df.loc[:, temp_test_df.columns != 'bug'])
    # unique_category_count = 2
    # inputs = tf.one_hot(temp_train_df['bug'], unique_category_count)
    # train_df['bug'] = inputs
    # val_df['bug'] = inputs
    # test_df['bug'] = inputs

    train_df['bug'] = temp_train_df['bug']
    val_df['bug'] = temp_val_df['bug']
    test_df['bug'] = temp_test_df['bug']

    return train_df, val_df, test_df

def classify(pred):
    ret = []
    for ele in pred:
        if ele<0.5:
            ret.append(0)
        else:
            ret.append(1)
    return ret

def confusionmetrics(pred, generator):
    y_pred = classify(pred.reshape(-1, 1))
    y_true = getreallabels(generator)

    cm = confusion_matrix(y_true, y_pred)
    return cm

def cmsave(cmlist, savepath, filenamestart, dataset):
    datapath = os.path.join(savepath, '{}_{}.{}'.format(filenamestart, dataset, 'csv'))
    df = pd.DataFrame(cmlist)
    df.to_csv(datapath,index=False)

def savehistory(name, history, argv, filename, input_widths, output_widths, batch):
    datapath = os.path.join(argv, '{}_{}_{}_{}_{}'.format(name, filename, batch, input_widths, output_widths))

    with open(datapath, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
    file_pi.close()

def plotmetrics(name, history, argv, filename, input_widths, output_widths, batch):
    datapath = os.path.join(argv, '{}_{}_{}_{}_{}.{}'.format(name, filename,batch,  input_widths, output_widths, 'pdf'))

    pp = PdfPages(datapath)
    fig = plt.figure()

    plt.subplot(411)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot mse during training
    plt.subplot(412)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.legend()

    plt.subplot(413)
    plt.title('auc')
    plt.plot(history.history['auc'], label='train')
    plt.plot(history.history['val_auc'], label='test')
    plt.legend()

    plt.subplot(414)
    plt.title('prc')
    plt.plot(history.history['prc'], label='train')
    plt.plot(history.history['val_prc'], label='test')
    plt.legend()

    pp.savefig(fig)
    pp.close()

def modeldataprepare(generator):
    hist_train, curr_train, label_train = [], [], []
    for hist, curr, labels in generator:
        for i in range(len(hist)):
            hist_train.append(hist[i])
            curr_train.append(curr[i])
            label_train.append(labels[i])
    return np.array(hist_train), np.array(curr_train), np.array(label_train)

def modeldataprepareWindowGenerator(generator):
    hist_train, label_train = [], []
    for hist, labels in generator:
        for i in range(len(hist)):
            hist_train.append(hist[i])
            label_train.append(labels[i])
    return np.array(hist_train), np.array(label_train)

def paramtunemetricsplot(argv, segnum, paramnum, history):
    datapath = os.path.join(argv, '{}_{}.{}'.format(segnum,paramnum, 'pdf'))

    pp = PdfPages(datapath)
    fig = plt.figure()

    plt.subplot(511)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    # plot mse during training
    plt.subplot(512)
    plt.title('auc')
    plt.plot(history.history['auc'], label='train')
    plt.plot(history.history['val_auc'], label='test')
    plt.legend()

    plt.subplot(513)
    plt.title('prc')
    plt.plot(history.history['prc'], label='train')
    plt.plot(history.history['val_prc'], label='test')
    plt.legend()

    plt.subplot(514)
    plt.title('precision')
    plt.plot(history.history['precision'], label='train')
    plt.plot(history.history['val_precision'], label='test')
    plt.legend()

    plt.subplot(515)
    plt.title('recall')
    plt.plot(history.history['recall'], label='train')
    plt.plot(history.history['val_recall'], label='test')
    plt.legend()


    pp.savefig(fig)
    pp.close()

def saveret(train_pred, train_real, name):
    ret_pred = []
    ret_true = []
    for i in range(len(train_pred)):
        ret_pred.append(train_pred[i][0])
        ret_true.append(train_real[i][0])
    df = pd.DataFrame([ret_pred, ret_true]).transpose()
    df.columns = ['pred', 'real']
    df.to_csv(name)

def paramtunemodelinitial(model, i, j, learning_rate,
                hist_train, curr_train, label_train,
                hist_val,curr_val,label_val,
                checkpoint_path,modelpath, predicpath, plotpath):
    model.compile(
        optimizer=tf.optimizers.Adam(lr=learning_rate),
        loss={
            "bugprone": tf.keras.losses.BinaryCrossentropy(),
            # "density": tf.keras.losses.MeanSquaredError(),
        },
        metrics=METRICS,
    )
    # checkpoint_path = os.path.join("C://Users//ZhaoY//Downloads//figures//chkpt", '{}_{}.{}'.format(i, i, 'ckpt'))
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor="val_prc",
                                                     save_best_only=True,
                                                     mode="max",
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit(
        {"histfea": hist_train, "currfea": curr_train},
        {"bugprone": label_train},
        epochs=300,
        batch_size=60,
        validation_data=({"histfea": hist_val, "currfea": curr_val}, {"bugprone": label_val}),
        shuffle=False,
        callbacks=[cp_callback]
    )
    paramtunemetricsplot(plotpath, i, j, history)
    # modelpath = os.path.join("C://Users//ZhaoY//Downloads//figures//chkpt", '{}.{}'.format(i, 'h5'))
    model.save(modelpath)
    model = load_model(modelpath)
    # model = load_model(modelpath, custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU, 'relu': tf.nn.relu})

    # model.load_weights(checkpoint_path)

    # Re-evaluate the model:
    loss_val, acc_val, pre_val, rec_val, auc_val, prc_val = model.evaluate(
        {"histfea": hist_val, "currfea": curr_val},
        {"bugprone": label_val},
        batch_size=60, verbose=2)
    f1 = 2*(pre_val*rec_val)/(pre_val+rec_val)
    dict = {'index': i, 'valloss': loss_val, 'valacc': acc_val, 'valpre': pre_val, 'valrec': rec_val, 'valauc': auc_val,
            'valprc': prc_val, 'f1':f1}

    pred = model.predict({"histfea": hist_val, "currfea": curr_val})
    # predicpath = os.path.join("C://Users//ZhaoY//Downloads//figures//chkpt", '{}_{}.{}'.format(i, 'prev', 'csv'))
    saveret(pred, label_val.reshape(-1, 1), predicpath)
    return dict, model

def lstm3layers(input, dropout):
    layer1 = tf.keras.layers.LSTM(128, return_sequences=True)(input)
    layer2 = tf.keras.layers.Dropout(dropout)(layer1)
    layer3 = tf.keras.layers.LSTM(64, return_sequences=True)(layer2)
    layer4 = tf.keras.layers.Dropout(dropout)(layer3)
    layer5 = tf.keras.layers.LSTM(32, return_sequences=False)(layer4)
    layer6 = tf.keras.layers.Dropout(dropout)(layer5)

    return layer6

def dense3layers(input, activation, regularizer, dropout):
    layer1 = tf.keras.layers.Dense(128, activity_regularizer=tf.keras.regularizers.L2(regularizer),
                                   activation=activation)(input)
    layer2 = tf.keras.layers.Dropout(dropout)(layer1)
    layer3 = tf.keras.layers.Dense(64, activity_regularizer=tf.keras.regularizers.L2(regularizer),
                                    activation=activation)(layer2)
    layer4 = tf.keras.layers.Dropout(dropout)(layer3)
    layer5 = tf.keras.layers.Dense(64, activity_regularizer=tf.keras.regularizers.L2(regularizer),
                                    activation=activation)(layer4)
    layer6 = tf.keras.layers.Dropout(dropout)(layer5)
    ret = tf.keras.layers.Dense(32, activity_regularizer=tf.keras.regularizers.L2(regularizer),
                                    activation=activation)(layer6)

    return ret

def modelcombinefeature(histwidth, currwidth, histsize, currsize, OUT_STEPS, output_bias, dropouthist, dropoutcurr, regularizer, activation):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    hist_input = tf.keras.Input(shape=(histwidth, histsize), name="histfea")
    curr_input = tf.keras.Input(shape=(currwidth, currsize), name="currfea")
    hist = lstm3layers(hist_input, dropouthist)
    curr = lstm3layers(curr_input, dropoutcurr)

    x = tf.keras.layers.concatenate([hist,curr])

    con = dense3layers(x, activation, regularizer, dropoutcurr)
    classes = tf.keras.layers.Dense(1,
                                   kernel_initializer=tf.initializers.zeros(),
                                   activity_regularizer=tf.keras.regularizers.L2(regularizer), activation='sigmoid',
                                   bias_initializer=output_bias, name="bugprone")(con)
    # densities = tf.keras.layers.Dense(1,
    #                                 kernel_initializer=tf.initializers.zeros(),
    #                                 activity_regularizer=tf.keras.regularizers.L2(regularizer), activation='linear',
    #                                 bias_initializer=output_bias, name="density")(x)
    model = tf.keras.Model([hist_input,curr_input], classes,)
    return model

def modelhistonly(histwidth, histsize, output_bias, dropouthist, regularizer, activation):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    hist_input = tf.keras.Input(shape=(histwidth, histsize), name="histfea")
    hist = lstm3layers(hist_input, dropouthist)

    con = dense3layers(hist, activation, regularizer,dropouthist)
    classes = tf.keras.layers.Dense(1,
                                   kernel_initializer=tf.initializers.zeros(),
                                   activity_regularizer=tf.keras.regularizers.L2(regularizer), activation='sigmoid',
                                   bias_initializer=output_bias, name="bugprone")(con)

    model = tf.keras.Model(hist_input, classes,)
    return model

def compile_and_fit(model, batch, maxepochs, checkpoint_path, lr, hist_train, label_train, hist_val, label_val):
    # early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
    #                                                 patience=patience,
    #                                                 mode='min')
# https://faroit.com/keras-docs/2.0.6/models/sequential/
    model.compile(loss='binary_crossentropy',   # how to choose
                optimizer=tf.optimizers.Adam(lr=lr),
                # sample_weight_mode="temporal",
                metrics=METRICS)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     monitor="val_prc",
                                                     save_best_only=True,
                                                     mode="max",
                                                     save_weights_only=True,
                                                     verbose=1)

    history = model.fit({"histfea": hist_train},
        {"bugprone": label_train},
        epochs=maxepochs,
        batch_size=batch,
        validation_data=({"histfea": hist_val}, {"bugprone": label_val}),
        shuffle=False,
        callbacks=[cp_callback])


    return history

class WGstruct():
    def __init__(self, input_width, label_width, shift,
                 sequence_stride, batch_size,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.sequence_stride = sequence_stride
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # self.hist_columns = {name: i for i, name in enumerate(self.column_indices) if
        #                 name not in ["bug","exp", "sexp", "fix", "ns", "lt"]}
        self.hist_columns = {name: i for i, name in enumerate(self.column_indices)}
        self.hist_column_indices = {name: i for i, name in enumerate(self.hist_columns)}

        self.curr_columns = train_df.loc[:, train_df.columns != "bug"].columns
        self.curr_column_indices= {name: i for i, name in enumerate(self.curr_columns)}

        # Work out the window parameters.
        self.hist_input_width = input_width - label_width
        self.curr_input_width = label_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.hist_input_slice = slice(0, self.total_window_size-self.label_width)
        self.hist_input_indices = np.arange(self.total_window_size)[self.hist_input_slice]

        self.label_start = self.total_window_size - self.label_width

        self.curr_input_slice = slice(self.label_start, None)
        self.curr_input_indices = np.arange(self.total_window_size)[self.curr_input_slice]

        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'history Input indices: {self.hist_input_indices}',
            f'history column name(s): {self.hist_columns}',
            f'current Input indices: {self.curr_input_indices}',
            f'current column name(s): {self.curr_columns}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        hist_inputs = features[:, self.hist_input_slice, :]
        curr_inputs = features[:, self.curr_input_slice, :]
        labels = features[:, self.labels_slice, :]

        # print(self.column_indices, self.column_indices['bug'])
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        curr_inputs = tf.stack(
            [curr_inputs[:, :, self.column_indices[name]] for name in self.curr_columns],
            axis=-1)
        hist_inputs = tf.stack(
            [hist_inputs[:, :, self.column_indices[name]] for name in self.hist_columns],
            axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        hist_inputs.set_shape([None, self.hist_input_width, None])
        curr_inputs.set_shape([None, self.curr_input_width, None])
        labels.set_shape([None, self.label_width, None])

        return hist_inputs, curr_inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=self.sequence_stride,
            shuffle=False,
            batch_size=self.batch_size, )


        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)