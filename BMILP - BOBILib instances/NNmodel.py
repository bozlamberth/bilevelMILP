import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math

tf.random.set_seed(1)

class NNmodel(object):
    def __init__(self, NNArchitecture, flag_ISNN, flag_doubleX):
        self.Architecture = NNArchitecture
        self.num_neuron_input = NNArchitecture[0]
        self.num_neuron_output = NNArchitecture[-1]
        self.num_neuron_hidden = NNArchitecture[1:-1]
        self.num_layer_hidden = len(self.num_neuron_hidden)
        self.flag_ISNN = flag_ISNN
        self.flag_doubleX = flag_doubleX
        if flag_ISNN:
            flag_ISNN = 'non_neg'
        else:
            flag_ISNN = None

        seednumber = 8
        z = [
            tf.keras.Input(shape=(self.num_neuron_input,))
        ]
        for i in range(self.num_layer_hidden):
            if i == 0:
                z += [
                    tf.keras.layers.Dense(self.num_neuron_hidden[i],
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seednumber),
                                          kernel_constraint=flag_ISNN,
                                          use_bias=True,
                                          bias_initializer='zeros',
                                          activation='sigmoid'
                                          )(z[i])
                ]
            else:
                z += [
                    tf.keras.layers.Dense(self.num_neuron_hidden[i],
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seednumber),
                                          kernel_constraint=flag_ISNN,
                                          use_bias=True,
                                          bias_initializer='zeros',
                                          activation='sigmoid'
                                          )(tf.keras.layers.concatenate([z[i], z[0]]))
                ]
        z += [
            tf.keras.layers.Dense(self.num_neuron_output,
                                  kernel_initializer=tf.keras.initializers.ones,
                                  kernel_constraint=flag_ISNN,
                                  use_bias=True,
                                  bias_initializer='zeros',
                                  activation='sigmoid'
                                  )(tf.keras.layers.concatenate([z[self.num_layer_hidden], z[0]]))
        ]
        self.model = tf.keras.Model(z[0], z[-1])
        self.model.summary()
        tf.keras.utils.plot_model(self.model, "NNmodel.png", show_shapes=True)

        self.history = None
        self.label_max = None
        self.label_min = None
        self.w = None
        self.b = None
        self.err_max = None
        self.err_min = None
        self.bs = None
        self.callback = None


    def train(self, data, num_epoch):
        def my_loss_upper(y_true, y_pre):
            loss1 = tf.square(y_true - y_pre) / 2
            loss2 = tf.maximum(y_true - y_pre, 0)
            return loss1 + loss2

        def my_loss_lower(y_true, y_pre):
            loss1 = tf.square(y_true - y_pre) / 2
            loss2 = tf.maximum(y_pre - y_true, 0)
            return loss1 + loss2

        def my_loss_policy(y_true, y_pre):
            err = tf.norm(y_true - y_pre, ord=2)
            loss = tf.square(err)
            return loss

        # data input
        num_x = len(data[1, :]) - self.num_neuron_output
        sample = data[:, 0:num_x]
        label = data[:, num_x:]
        if self.num_neuron_output == 1:
            self.label_max = np.max(label, axis=0)
            self.label_max = [self.label_max[i] if self.label_max[i] != 0 else 1 for i in range(len(label[0,:]))]
            self.label_min = np.min(label, axis=0)
            label_nor = (label - np.tile(self.label_min, (len(label[:,0]),1))) / (np.tile(self.label_max, (len(label[:,0]),1)) - np.tile(self.label_min, (len(label[:,0]),1))) * 1
        else:
            self.label_max = np.ones(self.num_neuron_output)
            self.label_min = np.zeros(self.num_neuron_output)
            label_nor = label

        # train NN
        self.bs = max(1, 2 ** (round(math.log2(len(label))) - 6))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01, decay=0.0001),
                           loss='binary_crossentropy',
                           # loss=my_loss_policy,
                           # loss=my_loss_upper,
                           # loss=my_loss_lower,
                           metrics=[tf.keras.metrics.MAE]  # , tf.keras.metrics.MAPE
                           )
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200, restore_best_weights=True, verbose=0)
        self.history = self.model.fit(sample if (not self.flag_doubleX) else np.hstack([sample, 1 - sample]), label_nor,
                                 epochs=num_epoch,
                                 batch_size=self.bs,
                                 verbose=1,
                                 validation_split=0.1,
                                 # validation_data=(x_val, y_val)
                                 callbacks=[self.callback]
                                 )
        self.showTraining()
        self.err_max, self.err_min = self.evaluate(sample, label)

        # save NN
        w = []
        b = []
        for i in range(int(len(self.model.weights) / 2)):
            temp = np.transpose(self.model.weights[2 * i].numpy())
            w.append(temp)
            temp = self.model.weights[2 * i + 1].numpy()
            b.append(temp.reshape(-1, 1))
        self.w = w
        self.b = b
        self.save()

        # # check approximation
        # temp = self.w[0] @ sample.T + self.b[0]
        # z = 1 - np.maximum(1 - np.maximum(temp / 5 + 0.5, 0), 0)
        # temp = self.w[1] @ np.vstack([z, sample.T]) + self.b[1]
        # z = 1 - np.maximum(1 - np.maximum(temp / 5 + 0.5, 0), 0)
        # temp = self.w[2] @ np.vstack([z, sample.T]) + self.b[2]
        # # z = 1 - np.maximum(1 - np.maximum(temp / 5 + 0.5, 0), 0)
        # # z = np.array([[np.int_(z[i, j] >= 0.5) for j in range(len(z[0, :]))] for i in range(len(z[:, 0]))])
        # z = temp
        # z = np.array([[np.int_(z[i, j] >= 0) for j in range(len(z[0, :]))] for i in range(len(z[:, 0]))])
        # temp = z.T - label
        # temp = np.int_(temp != 0)
        # print('approximation error of sigmoid:', sum(sum(temp)), '/', len(label[:,0]) * len(label[0,:]))
        return

    def train_continued(self, data, num_epoch):
        # data input
        num_x = len(data[1, :]) - self.num_neuron_output
        sample = data[:, 0:num_x]
        label = data[:, num_x:]
        label_nor = (label - self.label_min) / (self.label_max - self.label_min) * 1

        # train NN
        self.history = self.model.fit(sample if (not self.flag_doubleX) else np.hstack([sample, 1 - sample]), label_nor,
                                 epochs=num_epoch,
                                 batch_size=self.bs,
                                 verbose=0,
                                 validation_split=0.0,
                                 # validation_data=(x_val, y_val)
                                 callbacks=[self.callback]
                                 )
        # self.showTraining()
        # self.err_max, self.err_min = self.evaluate(sample, label)

        # save NN
        w = []
        b = []
        for i in range(int(len(self.model.weights) / 2)):
            temp = np.transpose(self.model.weights[2 * i].numpy())
            w.append(temp)
            temp = self.model.weights[2 * i + 1].numpy()
            b.append(temp.reshape(-1, 1))
        self.w = w
        self.b = b
        # self.save()
        return

    def save(self):
        FileName = 'NNparametersGNN.xlsx' if self.flag_ISNN == False else 'NNparametersISNN.xlsx'
        writer = pd.ExcelWriter(FileName)
        for i in range(int(len(self.model.weights) / 2)):
            wrt = pd.DataFrame(self.w[i])
            wrt.to_excel(writer, 'kernel' + str(i + 1), header=None, index=False)
            wrt = pd.DataFrame(self.b[i])
            wrt.to_excel(writer, 'bias' + str(i + 1), header=None, index=False)
        wrt = pd.DataFrame([self.Architecture, self.label_max, self.label_min, self.err_max, self.err_min])
        wrt.to_excel(writer, 'others', header=None, index=False)
        writer.close()
        # print('successfully export NN parameters as', FileName)
        return

    def showTraining(self):
        num_epoch = len(self.history.history['loss'])
        num_epoch_display = int(0.8 * num_epoch)
        # loss
        plt.figure()
        plt.plot(self.history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.ylabel('Model loss')
        plt.xlabel('Epoch')
        # plt.ylim(0, 1e-8)
        plt.ylim(0, self.history.history['loss'][-num_epoch_display])
        plt.legend(['Train_loss', 'Val_loss'])
        plt.show()
        # MAE
        plt.figure()
        plt.plot(self.history.history['mean_absolute_error'])
        # plt.plot(history.history['val_mean_absolute_error'])
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.ylim(0, self.history.history['mean_absolute_error'][-num_epoch_display])
        plt.legend(['Train_MAE', 'Val_MAE'])
        plt.show()
        return

    def evaluate(self, sample, label):
        phi_pre = self.model.predict(sample if (not self.flag_doubleX) else np.hstack([sample, 1-sample])) * (self.label_max - self.label_min) + self.label_min
        if len(label[0,:]) == 1:
            err = phi_pre - label
            err_max = np.max(err, axis=0)
            err_min = np.min(err, axis=0)
            err_relative = err / label * 100
            temp = np.hstack([phi_pre, label, err, err_relative])
            print('max_abs_error:', np.round(np.max(abs(err[:,0])), 2))
            print('max_relative_error:', np.round(np.max(abs(err_relative[:,0])), 2), '%')
            # absolute_error
            plt.figure()
            plt.boxplot(err[:,0])
            plt.ylabel('absolute_error')
            plt.show()
            # relative_error
            plt.figure()
            plt.boxplot(err_relative[:,0])
            plt.ylabel('relative_error (%)')
            plt.show()
        else:
            phi_pre = np.array([[np.int_(phi_pre[i,j] >= 0.5) for j in range(len(phi_pre[0,:]))] for i in range(len(phi_pre[:,0]))])
            # np.int_(np.round(phi_pre))
            err = np.sum(np.abs(phi_pre - label), axis=1)
            err_max = [np.max(err, axis=0)]
            err_min = [np.min(err, axis=0)]
            print('max_abs_error:', np.round(err_max, 2))
            # absolute_error
            plt.figure()
            plt.boxplot(err)
            plt.ylabel('absolute_error')
            plt.show()
        return err_max, err_min

    def readParameters(self, FileName):
        data = pd.read_excel(io=FileName, sheet_name=None, header=None)
        temp = data['others'].values[0]
        num_layer = sum(1 - np.isnan(data['others'].values[0]))
        self.Architecture = np.int_(temp[0:num_layer])
        self.num_layer_hidden = len(self.Architecture) - 2
        w = []
        b = []
        for i in range(self.num_layer_hidden + 1):
            w += [data['kernel' + str(i + 1)].values]
            b += [data['bias' + str(i + 1)].values]
        self.w = w
        self.b = b
        self.label_max = data['others'].values[1][0:self.num_neuron_output]
        self.label_min = data['others'].values[2][0:self.num_neuron_output]
        self.err_max = data['others'].values[3][0:self.num_neuron_output]
        self.err_min = data['others'].values[4][0:self.num_neuron_output]
        return

    def predict(self, x_temp):
        z = [x_temp]
        for i in range(self.num_layer_hidden + 1):
            if i == 0:
                temp = self.w[i] @ z[i] + self.b[i][:, 0]
                z += [np.maximum(temp, 0)]
            else:
                temp = self.w[i] @ np.hstack([z[i], z[0]]) + self.b[i][:, 0]
                if i != self.num_layer_hidden:
                    z += [np.maximum(temp, 0)]
                # else:
                #     z += [1 / (1 + np.exp(-temp))]
        return np.multiply(z[-1], self.label_max - self.label_min) + self.label_min

class NNmodel_simplified(object):
    def __init__(self):
        z = [
            tf.keras.Input(shape=(1,))
        ]
        z1 = tf.keras.layers.Dense(1,
                                      kernel_initializer=tf.keras.initializers.ones,
                                      kernel_constraint='non_neg',
                                      use_bias=True,
                                      bias_initializer='zeros',
                                      activation='sigmoid'
                                      )(z[0])
        z2 = tf.keras.layers.Dense(1,
                                      kernel_initializer=tf.keras.initializers.ones,
                                      kernel_constraint='non_neg',
                                      use_bias=True,
                                      bias_initializer='zeros',
                                      activation='relu'
                                      )(z[0])
        z += [
            tf.keras.layers.concatenate([z1, z2])
        ]
        z += [
            tf.keras.layers.Dense(1,
                                  kernel_initializer=tf.keras.initializers.ones,
                                  kernel_constraint='non_neg',
                                  use_bias=False
                                  )(z[1])
        ]
        self.model = tf.keras.Model(z[0], z[-1])
        self.model.summary()
        tf.keras.utils.plot_model(self.model, "NNmodel.png", show_shapes=True)

        self.history = None
        self.label_max = None
        self.label_min = None
        self.w = None
        self.b = None
        self.err_max = None
        self.err_min = None
        self.bs = None
        self.callback = None

        self.x0 = 0
        self.y0 = 1
        self.k1 = 1
        self.x1 = 1


    def train(self, data, num_epoch):
        # data input
        num_x = 1
        sample = data[:, 0:1]
        label = data[:, 1:]
        self.label_max = np.max(label, axis=0)
        self.label_min = np.min(label, axis=0)
        label_nor = (label - np.tile(self.label_min, (len(label[:,0]),1))) / (np.tile(self.label_max, (len(label[:,0]),1)) - np.tile(self.label_min, (len(label[:,0]),1))) * 1

        # train NN
        self.bs = max(1, 2 ** (round(math.log2(len(label))) - 6))
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, decay=0.001),
                           loss='mse',
                           metrics=[tf.keras.metrics.MAE]  # , tf.keras.metrics.MAPE
                           )
        self.callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=200, restore_best_weights=True, verbose=0)
        self.history = self.model.fit(sample, label_nor,
                                 epochs=num_epoch,
                                 batch_size=self.bs,
                                 verbose=1,
                                 validation_split=0.1,
                                 # validation_data=(x_val, y_val)
                                 callbacks=[self.callback]
                                 )
        self.showTraining()
        self.err_max, self.err_min = self.evaluate(sample, label)

        # save NN
        w = []
        b = []
        for i in range(int(np.ceil(len(self.model.weights) / 2))):
            temp = np.transpose(self.model.weights[2 * i].numpy())
            w.append(temp)
            if 2 * i + 1== len(self.model.weights):
                b.append(np.zeros((1,1)))
            else:
                temp = self.model.weights[2 * i + 1].numpy()
                b.append(temp.reshape(-1, 1))
        self.w = w
        self.b = b
        self.save()

        # check approximation
        temp = self.w[0] @ sample.T + self.b[0]
        z = 1 - np.maximum(1 - np.maximum(temp / 5 + 0.5, 0), 0)
        temp = self.w[1] @ np.vstack([z, sample.T]) + self.b[1]
        z = 1 - np.maximum(1 - np.maximum(temp / 5 + 0.5, 0), 0)
        temp = self.w[2] @ np.vstack([z, sample.T]) + self.b[2]
        # z = 1 - np.maximum(1 - np.maximum(temp / 5 + 0.5, 0), 0)
        # z = np.array([[np.int_(z[i, j] >= 0.5) for j in range(len(z[0, :]))] for i in range(len(z[:, 0]))])
        z = temp
        z = np.array([[np.int_(z[i, j] >= 0) for j in range(len(z[0, :]))] for i in range(len(z[:, 0]))])
        temp = z.T - label
        temp = np.int_(temp != 0)
        print('approximation error of sigmoid:', sum(sum(temp)), '/', len(label[:,0]) * len(label[0,:]))
        return

    def train_continued(self, data, num_epoch):
        # data input
        num_x = len(data[1, :]) - self.num_neuron_output
        sample = data[:, 0:num_x]
        label = data[:, num_x:]
        label_nor = (label - self.label_min) / (self.label_max - self.label_min) * 1

        # train NN
        self.history = self.model.fit(sample if (not self.flag_doubleX) else np.hstack([sample, 1 - sample]), label_nor,
                                 epochs=num_epoch,
                                 batch_size=self.bs,
                                 verbose=0,
                                 validation_split=0.0,
                                 # validation_data=(x_val, y_val)
                                 callbacks=[self.callback]
                                 )
        # self.showTraining()
        # self.err_max, self.err_min = self.evaluate(sample, label)

        # save NN
        w = []
        b = []
        for i in range(int(len(self.model.weights) / 2)):
            temp = np.transpose(self.model.weights[2 * i].numpy())
            w.append(temp)
            temp = self.model.weights[2 * i + 1].numpy()
            b.append(temp.reshape(-1, 1))
        self.w = w
        self.b = b
        # self.save()
        return

    def save(self):
        FileName = 'NNparametersGNN.xlsx' if self.flag_ISNN == False else 'NNparametersISNN.xlsx'
        writer = pd.ExcelWriter(FileName)
        for i in range(int(len(self.model.weights) / 2)):
            wrt = pd.DataFrame(self.w[i])
            wrt.to_excel(writer, 'kernel' + str(i + 1), header=None, index=False)
            wrt = pd.DataFrame(self.b[i])
            wrt.to_excel(writer, 'bias' + str(i + 1), header=None, index=False)
        wrt = pd.DataFrame([self.Architecture, self.label_max, self.label_min, self.err_max, self.err_min])
        wrt.to_excel(writer, 'others', header=None, index=False)
        writer.close()
        # print('successfully export NN parameters as', FileName)
        return

    def showTraining(self):
        num_epoch = len(self.history.history['loss'])
        num_epoch_display = int(0.8 * num_epoch)
        # loss
        plt.figure()
        plt.plot(self.history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.ylabel('Model loss')
        plt.xlabel('Epoch')
        # plt.ylim(0, 1e-8)
        plt.ylim(0, self.history.history['loss'][-num_epoch_display])
        plt.legend(['Train_loss', 'Val_loss'])
        plt.show()
        # MAE
        plt.figure()
        plt.plot(self.history.history['mean_absolute_error'])
        # plt.plot(history.history['val_mean_absolute_error'])
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.ylim(0, self.history.history['mean_absolute_error'][-num_epoch_display])
        plt.legend(['Train_MAE', 'Val_MAE'])
        plt.show()
        return

    def evaluate(self, sample, label):
        phi_pre = self.model.predict(sample) * (self.label_max - self.label_min) + self.label_min
        if len(label[0,:]) == 1:
            err = phi_pre - label
            err_max = np.max(err, axis=0)
            err_min = np.min(err, axis=0)
            err_relative = err / label * 100
            temp = np.hstack([phi_pre, label, err, err_relative])
            print('max_abs_error:', np.round(np.max(abs(err[:,0])), 2))
            print('max_relative_error:', np.round(np.max(abs(err_relative[:,0])), 2), '%')
            # absolute_error
            plt.figure()
            plt.boxplot(err[:,0])
            plt.ylabel('absolute_error')
            plt.show()
            # relative_error
            plt.figure()
            plt.boxplot(err_relative[:,0])
            plt.ylabel('relative_error (%)')
            plt.show()
        else:
            phi_pre = np.array([[np.int_(phi_pre[i,j] >= 0.5) for j in range(len(phi_pre[0,:]))] for i in range(len(phi_pre[:,0]))])
            # np.int_(np.round(phi_pre))
            err = np.sum(np.abs(phi_pre - label), axis=1)
            err_max = [np.max(err, axis=0)]
            err_min = [np.min(err, axis=0)]
            print('max_abs_error:', np.round(err_max, 2))
            # absolute_error
            plt.figure()
            plt.boxplot(err)
            plt.ylabel('absolute_error')
            plt.show()
        return err_max, err_min

    def readParameters(self, FileName):
        data = pd.read_excel(io=FileName, sheet_name=None, header=None)
        temp = data['others'].values[0]
        num_layer = sum(1 - np.isnan(data['others'].values[0]))
        self.Architecture = np.int_(temp[0:num_layer])
        self.num_layer_hidden = len(self.Architecture) - 2
        w = []
        b = []
        for i in range(self.num_layer_hidden + 1):
            w += [data['kernel' + str(i + 1)].values]
            b += [data['bias' + str(i + 1)].values]
        self.w = w
        self.b = b
        self.label_max = data['others'].values[1][0:self.num_neuron_output]
        self.label_min = data['others'].values[2][0:self.num_neuron_output]
        self.err_max = data['others'].values[3][0:self.num_neuron_output]
        self.err_min = data['others'].values[4][0:self.num_neuron_output]
        return

    def predict(self, x_temp):
        z = [x_temp]
        for i in range(self.num_layer_hidden + 1):
            if i == 0:
                temp = self.w[i] @ z[i] + self.b[i][:, 0]
                z += [np.maximum(temp, 0)]
            else:
                temp = self.w[i] @ np.hstack([z[i], z[0]]) + self.b[i][:, 0]
                if i != self.num_layer_hidden:
                    z += [np.maximum(temp, 0)]
                # else:
                #     z += [1 / (1 + np.exp(-temp))]
        return np.multiply(z[-1], self.label_max - self.label_min) + self.label_min


def calculateNumHidden(num_samples, num_layer, num_input, flag_ISNN):
    if flag_ISNN:
        return np.maximum(math.ceil((num_samples/(num_input+1)-1)/num_layer), 1)
    else:
        m = num_layer
        while m*(m+2)/4+(m+1)*(num_input+1) < num_samples:
            m += 1
        return np.maximum(math.ceil(m/num_layer), 1)


def encoder(x):
    num_x = x.shape[0]
    if num_x % 10 != 0:
        print('error number of x')
        return None
    num_seg = num_x // 10
    temp = np.zeros((10, num_x))
    for i in range(10):
        temp[i, i*num_seg:(i+1)*num_seg] = 2**np.array(range(num_seg))
    return temp @ x

