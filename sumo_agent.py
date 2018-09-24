# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

Interacting with traffic_light_dqn.py and map_computor.py

1) retriving values from sumo_computor.py

2) update state

3) controling logic

'''

from agent import State
from sys import platform
import sys
import os
import map_computor
import numpy as np
import shutil
import json


class Vehicles:
    initial_speed = 5.0

    def __init__(self):
        # add what ever you need to maintain
        self.id = None
        self.speed = None
        self.wait_time = None
        self.stop_count = None
        self.enter_time = None
        self.has_read = False
        self.first_stop_time = -1
        self.entering = True


class SumoAgent:

    class ParaSet:

        def __init__(self, dic_paras):
            for key, value in dic_paras.items():
                setattr(self, key, value)

    def __init__(self, sumo_cmd_str, path_set):

        self.path_set = path_set

        self.para_set = self.load_conf(os.path.join(self.path_set.PATH_TO_CONF, self.path_set.SUMO_AGENT_CONF))
        shutil.copy(
            os.path.join(self.path_set.PATH_TO_CONF, self.path_set.SUMO_AGENT_CONF),
            os.path.join(self.path_set.PATH_TO_OUTPUT, self.path_set.SUMO_AGENT_CONF))

        map_computor.start_sumo(sumo_cmd_str)

        self.dic_vehicles = {}
        self.state = None
        self.current_phase = 0
        self.current_phase_duration = 0

        self.update_state()
        self.update_vehicles()

        self.f_log_rewards = os.path.join(self.path_set.PATH_TO_OUTPUT, "log_rewards.txt")
        if not os.path.exists(self.f_log_rewards):
            f = open(self.f_log_rewards, 'w')
            list_reward_keys = np.sort(list(self.para_set.REWARDS_INFO_DICT.keys())+
                                       ['num_of_vehicles_in_system','num_of_vehicles_at_entering'])
            head_str = "count,action," + ','.join(list_reward_keys) + '\n'
            f.write(head_str)
            f.close()

    def end_sumo(self):
        map_computor.end_sumo()

    def load_conf(self, conf_file):
        dic_paras = json.load(open(conf_file, "r"))
        return self.ParaSet(dic_paras)

    def get_observation(self):
        return self.state

    def get_current_time(self):
        return map_computor.get_current_time()

    def get_current_phase(self):
        return self.current_phase

    def take_action(self, action):
        current_phase_number = self.get_current_phase()
        rewards_detail_dict_list = []
        if (self.current_phase_duration < self.para_set.MIN_PHASE_TIME[current_phase_number]):
            action = 0
        for i in range(self.para_set.MIN_ACTION_TIME):
            action_in_second = 0
            current_phase_number = self.get_current_phase()
            if action == 1 and i == 0:
                action_in_second = 1
            self.current_phase, self.current_phase_duration, self.vehicle_dict = map_computor.run(action=action_in_second,
                                                                               current_phase=current_phase_number,
                                                                               current_phase_duration=self.current_phase_duration,
                                                                               vehicle_dict=self.dic_vehicles,
                                                                               rewards_info_dict=self.para_set.REWARDS_INFO_DICT,
                                                                               f_log_rewards=self.f_log_rewards,
                                                                               rewards_detail_dict_list=rewards_detail_dict_list)  # run 1s SUMO

        #reward, reward_detail_dict = self.cal_reward(action)
        reward = self.cal_reward_from_list(rewards_detail_dict_list)
        #self.update_vehicles()
        self.update_state()

        return reward, action

    def take_action_pre_train(self, phase_time_now):
        current_phase_number = self.get_current_phase()
        rewards_detail_dict_list = []
        if (self.current_phase_duration < phase_time_now[current_phase_number]):
            action = 0
        else:
            action = 1
        for i in range(self.para_set.MIN_ACTION_TIME):
            action_in_second = 0
            current_phase_number = self.get_current_phase()
            if action == 1 and i == 0:
                action_in_second = 1
            self.current_phase, self.current_phase_duration, self.vehicle_dict = map_computor.run(action=action_in_second,
                                                                               current_phase=current_phase_number,
                                                                               current_phase_duration=self.current_phase_duration,
                                                                               vehicle_dict=self.dic_vehicles,
                                                                               rewards_info_dict=self.para_set.REWARDS_INFO_DICT,
                                                                               f_log_rewards=self.f_log_rewards,
                                                                               rewards_detail_dict_list=rewards_detail_dict_list)  # run 1s SUMO
        reward = self.cal_reward_from_list(rewards_detail_dict_list)

        #self.update_vehicles()
        self.update_state()

        return reward, action

    def update_vehicles(self):
        self.dic_vehicles = map_computor.update_vehicles_state(self.dic_vehicles)

    def update_state(self):
        status_tracker = map_computor.status_calculator()

        self.state = State(
            queue_length=np.reshape(np.array(status_tracker[0]), newshape=(1, 12)),
            num_of_vehicles=np.reshape(np.array(status_tracker[1]), newshape=(1, 12)),
            waiting_time=np.reshape(np.array(status_tracker[2]), newshape=(1, 12)),
            map_feature=np.reshape(np.array(status_tracker[3]), newshape=(1, 150, 150, 1)),
            cur_phase=np.reshape(np.array([self.current_phase]), newshape=(1, 1)),
            next_phase=np.reshape(np.array([(self.current_phase + 1) % len(self.para_set.MIN_PHASE_TIME)]), newshape=(1, 1)),
            time_this_phase=np.reshape(np.array([self.current_phase_duration]), newshape=(1, 1)),
            if_terminal=False
        )

    def cal_reward(self, action):
        # get directly from sumo
        reward, reward_detail_dict = map_computor.get_rewards_from_sumo(self.dic_vehicles, action, self.para_set.REWARDS_INFO_DICT)
        return reward*(1-0.8), reward_detail_dict

    def cal_reward_from_list(self, reward_detail_dict_list):
        reward = map_computor.get_rewards_from_dict_list(reward_detail_dict_list)
        return reward*(1-0.8)


if __name__ == '__main__':
    pass