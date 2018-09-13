import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import Input, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, Concatenate, GlobalAveragePooling2D, \
    BatchNormalization, AveragePooling2D, K
from keras.regularizers import l2
from keras_applications import densenet, resnet50
from keras_applications.densenet import backend, keras_utils
from keras_applications.imagenet_utils import _obtain_input_shape
from keras_applications.resnet50 import identity_block, conv_block
from keras_applications.vgg19 import WEIGHTS_PATH_NO_TOP, WEIGHTS_PATH, models, engine, layers
import os


class nerve_models(object):

    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def test_normol(self):
        inputs = Input(shape=(55,))

        # a layer instance is callable on a tensor, and returns a tensor
        x = Dense(64, activation='relu')(inputs)
        x = Dense(64, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)

        # This creates a model that includes
        # the Input layer and three Dense layers
        model = Model(inputs=inputs, outputs=predictions)
        return model

    def test_model(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), data_format='channels_last', padding='same', input_shape=self.input_shape))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', data_format='channels_first'))
        model.add(Activation('relu'))

        model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_first'))
        model.add(Dropout(0.25))

        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))

        return model

    def DenseNet(self, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=16, dropout_rate=None,
                       weight_decay=1E-4, verbose=True):
        '''
        denseNet 自定义结构
        :param depth:
        :param nb_dense_block:
        :param growth_rate:
        :param nb_filter:
        :param dropout_rate:
        :param weight_decay:
        :param verbose:
        :return:
        '''
        nb_classes = self.num_classes
        img_dim = self.input_shape
        def conv_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
            x = Activation('relu')(input)
            x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", use_bias=False,
                              kernel_regularizer=l2(weight_decay))(x)
            if dropout_rate is not None:
                x = Dropout(dropout_rate)(x)
            return x

        def transition_block(input, nb_filter, dropout_rate=None, weight_decay=1E-4):
            concat_axis = 1 if K.image_dim_ordering() == "th" else -1

            x = Convolution2D(nb_filter, (1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                              kernel_regularizer=l2(weight_decay))(input)
            if dropout_rate is not None:
                x = Dropout(dropout_rate)(x)
            x = AveragePooling2D((2, 2), strides=(2, 2))(x)

            x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                                   beta_regularizer=l2(weight_decay))(x)

            return x

        def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
            concat_axis = 1 if K.image_dim_ordering() == "th" else -1

            feature_list = [x]

            for i in range(nb_layers):
                x = conv_block(x, growth_rate, dropout_rate, weight_decay)
                feature_list.append(x)
                x = Concatenate(axis=concat_axis)(feature_list)
                nb_filter += growth_rate

            return x, nb_filter

        model_input = Input(shape=img_dim)

        concat_axis = 1 if K.image_dim_ordering() == "th" else -1

        assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

        # layers in each dense block
        nb_layers = int((depth - 4) / 3)

        # Initial convolution
        x = Convolution2D(nb_filter, (3, 3), kernel_initializer="he_uniform", padding="same", name="initial_conv2D",
                          use_bias=False,
                          kernel_regularizer=l2(weight_decay))(model_input)

        x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                               beta_regularizer=l2(weight_decay))(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block - 1):
            x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                       weight_decay=weight_decay)
            # add transition_block
            x = transition_block(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # The last dense_block does not have a transition_block
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)

        x = Activation('relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay),
                  bias_regularizer=l2(weight_decay))(x)

        densenet = Model(inputs=model_input, outputs=x)
        if verbose:
            print("DenseNet-%d-%d created." % (depth, growth_rate))
        return densenet

    def VGGCAM(self, num_input_channels=1024):
        """
        Build Convolution Neural Network
        args : nb_classes (int) number of classes
        returns : model (keras NN) the Neural Net model
        """
        nb_classes = self.num_classes
        model = Sequential()
        model.add(ZeroPadding2D((1, 1), input_shape=(self.input_shape), data_format='channels_last'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu'))

        # Add another conv layer with ReLU + GAP
        model.add(Conv2D(num_input_channels, (3, 3), activation='relu', padding="same"))
        model.add(AveragePooling2D((14, 14)))
        model.add(Flatten())
        # Add the W layer
        model.add(Dense(nb_classes, activation='softmax'))

        return model

    def VGG19(self):
        #还没测试通过
        classes = self.num_classes
        inputs = Input(shape=self.input_shape)
        # Block 1
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',

                          name='block1_conv1')(inputs)
        x = layers.Conv2D(64, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block1_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')(x)
        x = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv1')(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv2')(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv3')(x)
        x = layers.Conv2D(256, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block3_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv3')(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block4_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv1')(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv2')(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv3')(x)
        x = layers.Conv2D(512, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block5_conv4')(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)



        # Create model.
        model = models.Model(inputs, x, name='vgg19')


        return model

    def ResNet50(self):

        classes = self.num_classes
        img_input = Input(shape=self.input_shape)
        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
        x = layers.Conv2D(64, (7, 7),
                          strides=(2, 2),
                          padding='valid',
                          name='conv1')(x)

        x = layers.BatchNormalization(axis=3, name='bn_conv1')(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
        x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

        x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

        x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

        x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
        x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

        x = layers.AveragePooling2D((7, 7), name='avg_pool')(x)
        x = layers.Flatten()(x)

        x = layers.Dense(classes, activation='softmax', name='fc1000')(x)
        model = models.Model(img_input, x, name='resnet50')
        return model
