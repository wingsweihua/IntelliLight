# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

Controling agent, mainly choosing actions

'''

import json
import os
import shutil


class State(object):
    # ==========================

    D_QUEUE_LENGTH = (12,)
    D_NUM_OF_VEHICLES = (12,)
    D_WAITING_TIME = (12,)
    D_MAP_FEATURE = (150,150,1,)
    D_CUR_PHASE = (1,)
    D_NEXT_PHASE = (1,)
    D_TIME_THIS_PHASE = (1,)
    D_IF_TERMINAL = (1,)
    D_HISTORICAL_TRAFFIC = (6,)

    # ==========================

    def __init__(self,
                 queue_length, num_of_vehicles, waiting_time, map_feature,
                 cur_phase,
                 next_phase,
                 time_this_phase,
                 if_terminal):

        self.queue_length = queue_length
        self.num_of_vehicles = num_of_vehicles
        self.waiting_time = waiting_time
        self.map_feature = map_feature

        self.cur_phase = cur_phase
        self.next_phase = next_phase
        self.time_this_phase = time_this_phase

        self.if_terminal = if_terminal

        self.historical_traffic = None


class Agent(object):

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

            if hasattr(self, "STATE_FEATURE"):
                self.LIST_STATE_FEATURE = []
                list_state_feature_names = list(self.STATE_FEATURE.keys())
                list_state_feature_names.sort()
                for feature_name in list_state_feature_names:
                    if self.STATE_FEATURE[feature_name]:
                        self.LIST_STATE_FEATURE.append(feature_name)

    def __init__(self, num_phases,
                 path_set):

        self.path_set = path_set
        self.para_set = self.load_conf(os.path.join(self.path_set.PATH_TO_CONF, self.path_set.AGENT_CONF))
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.AGENT_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.AGENT_CONF))

        self.num_phases = num_phases
        self.state = None
        self.action = None
        self.memory = []
        self.average_reward = None

    def load_conf(self, conf_file):

        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def get_state(self, state, count):

        ''' set state for agent '''
        self.state = state
        return state

    def get_next_state(self, state, count):

        return state

    def choose(self, count, if_pretrain):

        ''' choose the best action for current state '''

        pass

    def remember(self, state, action, reward, next_state):
        ''' log the history separately '''

        pass

    def reset_update_count(self):

        pass

    def update_network(self, if_pretrain, use_average, current_time):
        pass

    def update_network_bar(self):
        pass

    def forget(self):
        pass

    def batch_predict(self,file_name="temp"):
        pass