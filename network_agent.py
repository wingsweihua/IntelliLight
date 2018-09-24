# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

Network agent from agent.py

'''


import numpy as np
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, Multiply, Add
from keras.models import Model, model_from_json, load_model
from keras.optimizers import RMSprop
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras import backend as K
import random
from keras.engine.topology import Layer
import os

from agent import Agent, State

class Selector(Layer):

    def __init__(self, select, **kwargs):
        super(Selector, self).__init__(**kwargs)
        self.select = select
        self.select_neuron = K.constant(value=self.select)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(Selector, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.cast(K.equal(x, self.select_neuron), dtype="float32")

    def get_config(self):
        config = {"select": self.select}
        base_config = super(Selector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def conv2d_bn(input_layer, index_layer,
              filters=16,
              kernel_size=(3, 3),
              strides=(1, 1)):
    """Utility function to apply conv + BN.
    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        name: name of the ops; will become `name + '_conv'`
            for the convolution and `name + '_bn'` for the
            batch norm layer.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  use_bias=False,
                  name="conv{0}".format(index_layer))(input_layer)
    bn = BatchNormalization(axis=bn_axis, scale=False, name="bn{0}".format(index_layer))(conv)
    act = Activation('relu', name="act{0}".format(index_layer))(bn)
    pooling = MaxPooling2D(pool_size=2)(act)
    x = Dropout(0.3)(pooling)
    return x


class NetworkAgent(Agent):

    @staticmethod
    def _unison_shuffled_copies(Xs, Y, sample_weight):
        p = np.random.permutation(len(Y))
        new_Xs = []
        for x in Xs:
            assert len(x) == len(Y)
            new_Xs.append(x[p])
        return new_Xs, Y[p], sample_weight[p]

    @staticmethod
    def _cnn_network_structure(img_features):
        conv1 = conv2d_bn(img_features, 1, filters=32, kernel_size=(8, 8), strides=(4, 4))
        conv2 = conv2d_bn(conv1, 2, filters=16, kernel_size=(4, 4), strides=(2, 2))
        img_flatten = Flatten()(conv2)
        return img_flatten

    @staticmethod
    def _shared_network_structure(state_features, dense_d):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_shared_1")(state_features)
        return hidden_1

    @staticmethod
    def _separate_network_structure(state_features, dense_d, num_actions, memo=""):
        hidden_1 = Dense(dense_d, activation="sigmoid", name="hidden_separate_branch_{0}_1".format(memo))(state_features)
        q_values = Dense(num_actions, activation="linear", name="q_values_separate_branch_{0}".format(memo))(hidden_1)
        return q_values

    def load_model(self, file_name):
        self.q_network = load_model(os.path.join(self.path_set.PATH_TO_MODEL, "%s_q_network.h5" % file_name))

    def save_model(self, file_name):
        self.q_network.save(os.path.join(self.path_set.PATH_TO_MODEL, "%s_q_network.h5" % file_name))

    def choose(self, count, if_pretrain):

        ''' choose the best action for current state '''

        q_values = self.q_network.predict(self.convert_state_to_input(self.state))
        # print(q_values)
        if if_pretrain:
            self.action = np.argmax(q_values[0])
        else:
            if random.random() <= self.para_set.EPSILON:  # continue explore new Random Action
                self.action = random.randrange(len(q_values[0]))
                print("##Explore")
            else:  # exploitation
                self.action = np.argmax(q_values[0])
            if self.para_set.EPSILON > 0.001 and count >= 20000:
                self.para_set.EPSILON = self.para_set.EPSILON * 0.9999
        return self.action, q_values

    def build_memory(self):

        return []

    def build_network_from_copy(self, network_copy):

        '''Initialize a Q network from a copy'''

        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={"Selector": Selector})
        network.set_weights(network_weights)
        network.compile(optimizer=RMSprop(lr=self.para_set.LEARNING_RATE),
                        loss="mean_squared_error")
        return network

    #
    # def batch_predict(self,file_name="temp"):
    #     f_samples = open("./records/DQN_v1/" + file_name[:file_name.rfind("_")] + "predict_pretrain.txt", "a")
    #     f_samples_head = ["state.cur_phase", "state.time_this_phase",
    #                       "target",
    #                       "action",
    #                       "reward"]
    #     f_samples.write('\t'.join(f_samples_head) + "\n")
    #     len_memory = len(self.memory)
    #     for i in range(len_memory):
    #         state, action, reward, next_state = self.memory[i]
    #         q_values = self.q_network.predict(self.convert_state_to_input(state))
    #         f_samples.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(
    #             str(state.cur_phase[0]), str(state.time_this_phase), str(q_values),
    #             str(action), str(reward)
    #         ))

    def remember(self, state, action, reward, next_state):

        ''' log the history '''
        self.memory.append([state, action, reward, next_state])

    def forget(self):

        ''' remove the old history if the memory is too large '''

        if len(self.memory) > self.para_set.MAX_MEMORY_LEN:
            print("length of memory: {0}, before forget".format(len(self.memory)))
            self.memory = self.memory[-self.para_set.MAX_MEMORY_LEN:]
            print("length of memory: {0}, after forget".format(len(self.memory)))

    def _get_next_estimated_reward(self, next_state):

        if self.para_set.DDQN:
            a_max = np.argmax(self.q_network.predict(
                self.convert_state_to_input(next_state))[0])
            next_estimated_reward = self.q_network_bar.predict(
                self.convert_state_to_input(next_state))[0][a_max]
            return next_estimated_reward
        else:
            next_estimated_reward = np.max(self.q_network_bar.predict(
                self.convert_state_to_input(next_state))[0])
            return next_estimated_reward

    def update_network_bar(self):

        ''' update Q bar '''

        if self.q_bar_outdated >= self.para_set.UPDATE_Q_BAR_FREQ:
            self.q_network_bar = self.build_network_from_copy(self.q_network)
            self.q_bar_outdated = 0