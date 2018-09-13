import os
import pickle
import keras
import numpy as np
from PIL import Image
from keras.engine.saving import load_model
from keras.optimizers import Adam
from myTools.pdTools.auto_keras_tool.keras_models import nerve_models


class Aktl(object):
    '''
    keras图像自动化处理工具
    基于tensorflow—GPU的keras框架
    导入一个文件夹，包含三个目录train文件夹，放训练图片，test文件夹，放测试图片，Target_dictionary.txt 文件
    分别是训练图片的名字（包括结尾格式）和分类名（可以是文本型）
    版本1.2 区分图片模式和自定义矩阵模式，默认图片模式
    '''
    def __init__(self, data_path, mode_name, load_mode='RAM', img_convert='RGB', deal_mode = 'img'):
        '''

        :param data_path: 总文件夹的路径
        :param load_mode: 加载方式，默认全部加载到显存'RAM'，如果显存不足应该选择'DISK'
        :param img_convert L,RGB，RGBA三种读取方式
        :param deal_mode: "img"图像模式，"any"任意的数据矩阵
        '''
        self.deal_mode = deal_mode
        self.data_path = data_path
        self.mode_name = mode_name
        self.data_load()                    #载入数据
        self.mode_path = self.mode_name + '.h5'
        self.load_mode = load_mode
        self.img_convert = img_convert
        self.data_list()                    #生成参数目录
        self.data_prepare()                 #数据预处理
        self.data_conversion()
        if self.load_mode == 'RAM':
            self.RAM_load()
        elif self.load_mode == 'DISK':
            self.DISK_prepare()
            self.DISK_load()


    def data_load(self):
        train_path = os.path.join(self.data_path, 'train')
        test_path = os.path.join(self.data_path, 'test')
        txt_path = os.path.join(self.data_path, 'Target_dictionary.txt')

        self.filename_dataset = {}          #{'test': ['E:\\test_img\\test\\1920x1200.jpg']}
        self.filename_dataset['train'] = []
        self.filename_dataset['test'] = []
        self.target = {}                    #{'E:\\test_img\\train\\train3.jpg': 'grass',
        for i in os.listdir(train_path):
            self.filename_dataset['train'].append(train_path + '\\' + i)

        for root, dirs, files in os.walk(test_path):
            if not files:
                continue
            for i in files:
                self.filename_dataset['test'].append(root + '\\' + i)

        with open(txt_path) as file_read:
            for i in file_read:
                i = i.replace("\n", '')
                taerget_split = i.split("\t")
                if i != '':
                    self.target[os.path.join(train_path, taerget_split[0])] =taerget_split[1]

    def DISK_prepare(self):
        img = Image.open(self.X_train_temp[0]).convert(self.img_convert)
        img = np.array(img)
        self.input_shape = img.shape

    def data_list(self):
        self.X_train_temp = []              #图片名列表
        self.Y_train_temp = []              #1，2，3数字列表
        self.X_train = []
        self.Y_train = []
        self.targer_to_num_dir = {}         #{'water': 3, 'tree': 1, 'grass': 0, 'sun': 2}

    def data_prepare(self):
        ttd_temp = 0
        for i in self.target:
            if self.target[i] not in self.targer_to_num_dir:
                self.targer_to_num_dir[self.target[i]] = ttd_temp
                ttd_temp += 1
        for i in self.filename_dataset['train']:
            self.X_train_temp.append(i)
            self.Y_train_temp.append(self.targer_to_num_dir[self.target[i]])
        self.num_classes = len(self.targer_to_num_dir)


    def RAM_load(self):
        x_train = []
        for i in self.X_train_temp:
            img = Image.open(i).convert(self.img_convert)
            imgn = np.array(img)
            x_train.append(imgn)

        self.X_train = np.array(x_train)/255
        self.Y_train = keras.utils.to_categorical(self.Y_train_temp, len(self.targer_to_num_dir))
        self.input_shape = self.X_train.shape[-3:]  # 分类数量

    def DISK_load(self):
        for x, y in zip(self.X_train_temp, self.Y_train_temp):
            x_train = []
            y_train = []
            img = Image.open(x).convert(self.img_convert)
            img = np.array(img)
            x_train.append(img)
            x_train = np.array(x_train) / 255
            y_train.append(y)
            y_train = keras.utils.to_categorical(y_train, len(self.targer_to_num_dir))
            yield (x_train, y_train)


    def data_conversion(self):
        #原始数据格式处理
        pass

    def DenseNet(self, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                                                 weight_decay=1E-4, verbose=True):
        #比较灵活的DenseNet模型
        self.model = nerve_models(self.input_shape, self.num_classes).DenseNet(depth=depth,
                                                                               nb_dense_block=nb_dense_block,
                                                                               growth_rate=growth_rate,
                                                                               nb_filter=nb_filter,
                                                                               dropout_rate=dropout_rate,
                                                                               weight_decay=weight_decay,
                                                                               verbose=verbose)

        opt = Adam()
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        return self.model

    def VGGCAM(self, num_input_channels=1024):

        self.model = nerve_models(self.input_shape, self.num_classes).VGGCAM(num_input_channels)
        self.model.compile(optimizer="sgd", loss='categorical_crossentropy')
        return self.model

    def test_model(self):
        #最简单的测试model
        self.model = nerve_models(self.input_shape, self.num_classes).test_model()
        self.model.compile(optimizer="sgd", loss='categorical_crossentropy')
        return self.model

    def ResNet50(self):
        #最简单的测试model
        self.model = nerve_models(self.input_shape, self.num_classes).ResNet50()
        self.model.compile(optimizer="sgd", loss='categorical_crossentropy')
        return self.model

    def fit(self, batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None,
            validation_split=0.,
            validation_data=None,
            shuffle=True,
            class_weight=None,
            sample_weight=None,
            initial_epoch=0,
            steps_per_epoch=None,
            validation_steps=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False):

        if os.path.exists(self.mode_path):
            self.model = load_model(self.mode_path)
            print('由模型{0}继续训练'.format(self.mode_path))

        if self.load_mode == 'RAM':
            print('将所有数据加入内存进行训练，内存不够可换为DISK模式')
            self.model.fit(self.X_train, self.Y_train, batch_size=batch_size,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        shuffle=shuffle,
                        class_weight=class_weight,
                        sample_weight=sample_weight,
                        initial_epoch=initial_epoch,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps)

        elif self.load_mode == 'DISK':
            print('将从磁盘单独加载训练数据')
            self.model.fit_generator(self.DISK_load(),
                    steps_per_epoch=len(self.X_train_temp),
                    epochs=epochs,
                    verbose=verbose,
                    callbacks=callbacks,
                    validation_data=validation_data,
                    validation_steps=validation_steps,
                    class_weight=class_weight,
                    max_queue_size=max_queue_size,
                    workers=workers,
                    use_multiprocessing=use_multiprocessing,
                    shuffle=shuffle,
                    initial_epoch=initial_epoch)

        self.model.save(self.mode_path)
        model_save = self.filename_dataset, self.targer_to_num_dir, self.img_convert

        pickle.dump(model_save, open('model_param.pickle', 'wb'))


class Aktl_predict(object):
    '''
    配套预测类
    '''
    def __init__(self, model_path, param_path):
        self.model_path = model_path
        self.param_path = param_path
        self.param_loal()
        self.target_deal()
        self.moder_predict()

    def param_loal(self):
        self.filename_dataset, self.targer_to_num_dir, self.img_convert\
            = pickle.load(open(self.param_path, mode='rb'))

        self.model = load_model(self.model_path)

    def target_deal(self):
        self.target = {}                                    #输出值的对应字典，可直接对位分类结果
        for i in self.targer_to_num_dir:
            self.target[self.targer_to_num_dir[i]] = i

    def moder_predict(self):
        x_test = []
        for i in self.filename_dataset['test']:
            img = Image.open(i).convert(self.img_convert)
            imgn = np.array(img)
            x_test.append(imgn)
        self.X_train = np.array(x_test)/255
        self.result = self.model.predict(self.X_train)

'''
训练demo
data_path = r'E:\test_img'
asp = Aktl(data_path, '62', load_mode='DISK')
asp.model = asp.test_model()
asp.fit()
'''

'''
测试demo
from myTools.pdTools.auto_keras_tool.akt_1_1 import Aktl_predict
model_name = '62.h5'
param_path = 'model_param.pickle'
asp = Aktl_predict(model_name, param_path)
print(asp.target)
print(asp.result)
'''







