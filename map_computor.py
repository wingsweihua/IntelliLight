# -*- coding: utf-8 -*-

'''
@author: hzw77, gjz5038

1) interacting with SUMO, including
      retrive values, set lights
2) interacting with sumo_agent, including
      returning status, rewards, etc.

'''

import numpy as np
import math
import os
import sys
import xml.etree.ElementTree as ET
from sys import platform
from sumo_agent import Vehicles

###### Please Specify the location of your traci module

if platform == "linux" or platform == "linux2":# this is linux
    os.environ['SUMO_HOME'] = '/usr/share/sumo'
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
	            os.path.join(os.environ["SUMO_HOME"], "tools")
	        )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == "win32":
    os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\DLR\\Sumo'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/sumo-git".format(os.getlogin())

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

else:
    sys.exit("platform error")

yeta = 0.15
tao = 2
constantC = 40.0
carWidth = 3.3
grid_width = 4
area_length = 600
direction_lane_dict = {"NSG": [1, 0], "SNG": [1, 0], "EWG": [1, 0], "WEG": [1, 0],
                       "NWG": [0], "WSG": [0], "SEG": [0], "ENG": [0],
                       "NEG": [2], "WNG": [2], "SWG": [2], "ESG": [2]}
direction_list = ["NWG", "WSG", "SEG", "ENG", "NSG", "SNG", "EWG", "WEG", "NEG", "WNG", "SWG", "ESG"]

#min_phase_time = [30, 96, 74]
min_phase_time_7 = [10, 35]
node_light_7 = "node0"
phases_light_7 = ["WNG_ESG_EWG_WEG_WSG_ENG", "NSG_NEG_SNG_SWG_NWG_SEG"]
WNG_ESG_EWG_WEG_WSG_ENG = "grrr gGGG grrr gGGG".replace(" ", "")
NSG_NEG_SNG_SWG_NWG_SEG = "gGGG grrr gGGG grrr".replace(" ", "")
controlSignal = (WNG_ESG_EWG_WEG_WSG_ENG, NSG_NEG_SNG_SWG_NWG_SEG)

listLanes=['edge1-0_0','edge1-0_1','edge1-0_2','edge2-0_0','edge2-0_1','edge2-0_2',
                                 'edge3-0_0','edge3-0_1','edge3-0_2','edge4-0_0','edge4-0_1','edge4-0_2']
'''
input: phase "NSG_SNG" , four lane number, in the key of W,E,S,N
output: 
1.affected lane number: 4_0_0, 4_0_1, 3_0_0, 3_0_1
# 2.destination lane number, 0_3_0,0_3_1  

'''


def start_sumo(sumo_cmd_str):
    traci.start(sumo_cmd_str)
    for i in range(20):
        traci.simulationStep()

def end_sumo():
    traci.close()

def get_current_time():
    return traci.simulation.getCurrentTime() / 1000

def phase_affected_lane(phase="NSG_SNG",
                        four_lane_ids={'W': 'edge1-0', "E": "edge2-0", 'S': 'edge4-0', 'N': 'edge3-0'}):
    directions = phase.split('_')
    affected_lanes = []
    for direction in directions:
        for k, v in four_lane_ids.items():
            if v.strip() != '' and direction.startswith(k):
                for lane_no in direction_lane_dict[direction]:
                    affected_lanes.append("%s_%d" % (v, lane_no))
                    # affacted_lanes.append("%s_%d" % (v, 0))
    if affected_lanes == []:
        raise("Please check your phase and lane_number_dict in phase_affacted_lane()!")
    return affected_lanes


'''
input: central nodeid "node0", surrounding nodes WESN: [1,2,3,4]
output: four_lane_ids={'W':'edge1-0',"E":"edge2-0",'S':'edge4-0','N':'edge3-0'})
'''


def find_surrounding_lane_WESN(central_node_id="node0", WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    tree = ET.parse('./data/cross.net.xml')
    root = tree.getroot()
    four_lane_ids_dict = {}
    for k, v in WESN_node_ids.items():
        four_lane_ids_dict[k] = root.find("./edge[@from='%s'][@to='%s']" % (v, central_node_id)).get('id')
    return four_lane_ids_dict


'''
coordinate mapper
'''


def coordinate_mapper(x1, y1, x2, y2, area_length=600, area_width=600):
    x1 = int(x1 / grid_width)
    y1 = int(y1 / grid_width)
    x2 = int(x2 / grid_width)
    y2 = int(y2 / grid_width)
    x_max = x1 if x1 > x2 else x2
    x_min = x1 if x1 < x2 else x2
    y_max = y1 if y1 > y2 else y2
    y_min = y1 if y1 < x2 else y2
    length_num_grids = int(area_length / grid_width)
    width_num_grids = int(area_width / grid_width)
    return length_num_grids - y_max, length_num_grids - y_min, x_min, x_max

def get_phase_affected_lane_traffic_max_volume(phase="NSG_SNG", tl_node_id="node0",
                                 WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    directions = phase.split('_')
    traffic_volume_start_end = []
    for direction in directions:
        traffic_volume_start_end.append([four_lane_ids_dict[direction[0]],four_lane_ids_dict[direction[1]]])
    tree = ET.parse('./data/cross.rou.xml')
    root = tree.getroot()
    phase_volumes = []
    for lane_id in traffic_volume_start_end:
        to_lane_id="edge%s-%s"%(lane_id[1].split('-')[1],lane_id[1].split('-')[0][4:])
        time_begin = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('begin')
        time_end = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('end')
        volume = root.find("./flow[@from='%s'][@to='%s']" % (lane_id[0], to_lane_id)).get('number')
        phase_volumes.append((float(time_end)-float(time_begin))/float(volume))
    return max(phase_volumes)


def phase_affected_lane_position(phase="NSG_SNG", tl_node_id="node0",
                                 WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    '''
    input: NSG_SNG ,central nodeid "node0", surrounding nodes WESN: {"W":"1", "E":"2", "S":"3", "N":"4"}
    output: edge-ids, 4_0_0, 4_0_1, 3_0_0, 3_0_1
    [[ 98,  100,  204,  301],[ 102, 104, 104, 198]]
    '''
    four_lane_ids_dict = find_surrounding_lane_WESN(central_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    affected_lanes = phase_affected_lane(phase=phase, four_lane_ids=four_lane_ids_dict)
    tree = ET.parse('./data/cross.net.xml')
    root = tree.getroot()
    indexes = []
    for lane_id in affected_lanes:
        lane_shape = root.find("./edge[@to='%s']/lane[@id='%s']" % (tl_node_id, lane_id)).get('shape')
        lane_x1 = float(lane_shape.split(" ")[0].split(",")[0])
        lane_y1 = float(lane_shape.split(" ")[0].split(",")[1])
        lane_x2 = float(lane_shape.split(" ")[1].split(",")[0])
        lane_y2 = float(lane_shape.split(" ")[1].split(",")[1])
        ind_x1, ind_x2, ind_y1, ind_y2 = coordinate_mapper(lane_x1, lane_y1, lane_x2, lane_y2)
        indexes.append([ind_x1, ind_x2 + 1, ind_y1, ind_y2 + 1])
    return indexes


def phases_affected_lane_postions(phases=["NSG_SNG_NWG_SEG", "NEG_SWG_NWG_SEG"], tl_node_id="node0",
                                  WESN_node_ids={"W": "1", "E": "2", "S": "3", "N": "4"}):
    parameterArray = []
    for phase in phases:
        parameterArray += phase_affected_lane_position(phase=phase, tl_node_id=tl_node_id, WESN_node_ids=WESN_node_ids)
    return parameterArray


def vehicle_location_mapper(coordinate, area_length=600, area_width=600):
    transformX = math.floor(coordinate[0] / grid_width)
    transformY = math.floor((area_length - coordinate[1]) / grid_width)
    length_num_grids = int(area_length/grid_width)
    transformY = length_num_grids-1 if transformY == length_num_grids else transformY
    transformX = length_num_grids-1 if transformX == length_num_grids else transformX
    tempTransformTuple = (transformY, transformX)
    return tempTransformTuple


def translateAction(action):
    result = 0
    for i in range(len(action)):
        result += (i + 1) * action[i]
    return result




def changeTrafficLight_7(current_phase=0):  # [WNG_ESG_WSG_ENG_NWG_SEG]
    # phases=["WNG_ESG_WSG_ENG_NWG_SEG","EWG_WEG_WSG_ENG_NWG_SEG","NSG_NEG_SNG_SWG_WSG_ENG_NWG_SEG"]
    next_phase = (current_phase + 1) % len(controlSignal)
    next_phase_time_eclipsed = 0
    traci.trafficlights.setRedYellowGreenState(node_light_7, controlSignal[next_phase])
    return next_phase, next_phase_time_eclipsed




def get_phase_vector(current_phase=0):
    phase = phases_light_7[current_phase].split("_")
    phase_vector = [0] * len(direction_list)
    for direction in phase:
        phase_vector[direction_list.index(direction)] = 1
    return np.array(phase_vector)


def getMapOfCertainTrafficLight(curtent_phase=0, tl_node_id="node0", area_length=600):
    current_phases_light_7 = [phases_light_7[curtent_phase]]
    parameterArray = phases_affected_lane_postions(phases=current_phases_light_7)
    length_num_grids = int(area_length / grid_width)
    resultTrained = np.zeros((length_num_grids, length_num_grids))
    for affected_road in parameterArray:
        resultTrained[affected_road[0]:affected_road[1], affected_road[2]:affected_road[3]] = 127
    return resultTrained




def calculate_reward(tempLastVehicleStateList):
    waitedTime = 0
    stop_count = 0
    for key, vehicle_dict in tempLastVehicleStateList.items():
        if tempLastVehicleStateList[key]['speed'] < 5:
            waitedTime += 1
            #waitedTime += (1 +math.pow(tempLastVehicleStateList[key]['waitedTime']/50,2))
        if tempLastVehicleStateList[key]['former_speed'] > 0.5 and tempLastVehicleStateList[key]['speed'] < 0.5:
            stop_count += (tempLastVehicleStateList[key]['stop_count']-tempLastVehicleStateList[key]['former_stop_count'])
    #PI = (waitedTime + 10 * stop_count) / len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    PI = waitedTime/len(tempLastVehicleStateList) if len(tempLastVehicleStateList)!=0 else 0
    return - PI

def getMapOfVehicles(area_length=600):
    '''
    get the vehicle positions as NIPS paper
    :param area_length:
    :return: numpy narray
    '''
    length_num_grids = int(area_length / grid_width)
    mapOfCars = np.zeros((length_num_grids, length_num_grids))

    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        vehicle_position = traci.vehicle.getPosition(vehicle_id)  # (double,double),tuple

        transform_tuple = vehicle_location_mapper(vehicle_position)  # call the function
        mapOfCars[transform_tuple[0], transform_tuple[1]] = 1

    return mapOfCars

def restrict_reward(reward,func="unstrict"):
    if func == "linear":
        bound = -50
        reward = 0 if reward < bound else (reward/(-bound) + 1)
    elif func == "neg_log":
        reward = math.log(-reward+1)
    else:
        pass

    return reward


def log_rewards(vehicle_dict, action, rewards_info_dict, file_name, timestamp,rewards_detail_dict_list):

    reward, reward_detail_dict = get_rewards_from_sumo(vehicle_dict, action, rewards_info_dict)
    list_reward_keys = np.sort(list(reward_detail_dict.keys()))
    reward_str = "{0}, {1}".format(timestamp,action)
    for reward_key in list_reward_keys:
        reward_str = reward_str + ", {0}".format(reward_detail_dict[reward_key][2])
    reward_str += '\n'

    fp = open(file_name, "a")
    fp.write(reward_str)
    fp.close()
    rewards_detail_dict_list.append(reward_detail_dict)


def get_rewards_from_sumo(vehicle_dict, action, rewards_info_dict,
                          listLanes=['edge1-0_0','edge1-0_1','edge1-0_2','edge2-0_0','edge2-0_1','edge2-0_2',
                                 'edge3-0_0','edge3-0_1','edge3-0_2','edge4-0_0','edge4-0_1','edge4-0_2'],):
    reward = 0
    import copy
    reward_detail_dict = copy.deepcopy(rewards_info_dict)

    vehicle_id_entering_list = get_vehicle_id_entering()

    reward_detail_dict['queue_length'].append(get_overall_queue_length(listLanes))
    reward_detail_dict['wait_time'].append(get_overall_waiting_time(listLanes))
    reward_detail_dict['delay'].append(get_overall_delay(listLanes))
    reward_detail_dict['emergency'].append(get_num_of_emergency_stops(vehicle_dict))
    reward_detail_dict['duration'].append(get_travel_time_duration(vehicle_dict, vehicle_id_entering_list))
    reward_detail_dict['flickering'].append(get_flickering(action))
    reward_detail_dict['partial_duration'].append(get_partial_travel_time_duration(vehicle_dict, vehicle_id_entering_list))

    vehicle_id_list = traci.vehicle.getIDList()
    reward_detail_dict['num_of_vehicles_in_system'] = [False, 0, len(vehicle_id_list)]

    reward_detail_dict['num_of_vehicles_at_entering'] = [False, 0, len(vehicle_id_entering_list)]


    vehicle_id_leaving = get_vehicle_id_leaving(vehicle_dict)

    reward_detail_dict['num_of_vehicles_left'].append(len(vehicle_id_leaving))
    reward_detail_dict['duration_of_vehicles_left'].append(get_travel_time_duration(vehicle_dict, vehicle_id_leaving))



    for k, v in reward_detail_dict.items():
        if v[0]:  # True or False
            reward += v[1]*v[2]
    reward = restrict_reward(reward)#,func="linear")
    return reward, reward_detail_dict

def get_rewards_from_dict_list(rewards_detail_dict_list):
    reward = 0
    for i in range(len(rewards_detail_dict_list)):
        for k, v in rewards_detail_dict_list[i].items():
            if v[0]:  # True or False
                reward += v[1] * v[2]
    reward = restrict_reward(reward)
    return reward

def get_overall_queue_length(listLanes):
    overall_queue_length = 0
    for lane in listLanes:
        overall_queue_length += traci.lane.getLastStepHaltingNumber(lane)
    return overall_queue_length

def get_overall_waiting_time(listLanes):
    overall_waiting_time = 0
    for lane in listLanes:
        overall_waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0
    return overall_waiting_time

def get_overall_delay(listLanes):
    overall_delay = 0
    for lane in listLanes:
        overall_delay += 1 - traci.lane.getLastStepMeanSpeed(str(lane)) / traci.lane.getMaxSpeed(str(lane))
    return overall_delay

def get_flickering(action):
    return action

# calculate number of emergency stops by vehicle
def get_num_of_emergency_stops(vehicle_dict):
    emergency_stops = 0
    vehicle_id_list = traci.vehicle.getIDList()
    for vehicle_id in vehicle_id_list:
        traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
        current_speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
        if (vehicle_id in vehicle_dict.keys()):
            vehicle_former_state = vehicle_dict[vehicle_id]
            if current_speed - vehicle_former_state.speed < -4.5:
                emergency_stops += 1
        else:
            # print("##New car coming")
            if current_speed - Vehicles.initial_speed < -4.5:
                emergency_stops += 1
    if len(vehicle_dict) > 0:
        return emergency_stops/len(vehicle_dict)
    else:
        return 0

def get_partial_travel_time_duration(vehicle_dict, vehicle_id_list):
    travel_time_duration = 0
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()) and (vehicle_dict[vehicle_id].first_stop_time != -1):
            travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[vehicle_id].first_stop_time)/60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration#/len(vehicle_id_list)
    else:
        return 0


def get_travel_time_duration(vehicle_dict, vehicle_id_list):
    travel_time_duration = 0
    for vehicle_id in vehicle_id_list:
        if (vehicle_id in vehicle_dict.keys()):
            travel_time_duration += (traci.simulation.getCurrentTime() / 1000 - vehicle_dict[vehicle_id].enter_time)/60.0
    if len(vehicle_id_list) > 0:
        return travel_time_duration#/len(vehicle_id_list)
    else:
        return 0

def update_vehicles_state(dic_vehicles):
    vehicle_id_list = traci.vehicle.getIDList()
    vehicle_id_entering_list = get_vehicle_id_entering()
    for vehicle_id in (set(dic_vehicles.keys())-set(vehicle_id_list)):
        del(dic_vehicles[vehicle_id])

    for vehicle_id in vehicle_id_list:
        if (vehicle_id in dic_vehicles.keys()) == False:
            vehicle = Vehicles()
            vehicle.id = vehicle_id
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_LANE_ID, tc.VAR_SPEED))
            vehicle.speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
            current_sumo_time = traci.simulation.getCurrentTime()/1000
            vehicle.enter_time = current_sumo_time
            # if it enters and stops at the very first
            if (vehicle.speed < 0.1) and (vehicle.first_stop_time == -1):
                vehicle.first_stop_time = current_sumo_time
            dic_vehicles[vehicle_id] = vehicle
        else:
            dic_vehicles[vehicle_id].speed = traci.vehicle.getSubscriptionResults(vehicle_id).get(64)
            if (dic_vehicles[vehicle_id].speed < 0.1) and (dic_vehicles[vehicle_id].first_stop_time == -1):
                dic_vehicles[vehicle_id].first_stop_time = traci.simulation.getCurrentTime()/1000
            if (vehicle_id in vehicle_id_entering_list) == False:
                dic_vehicles[vehicle_id].entering = False

    return dic_vehicles

def status_calculator():
    laneQueueTracker=[]
    laneNumVehiclesTracker=[]
    laneWaitingTracker=[]
    #================= COUNT HALTED VEHICLES (I.E. QUEUE SIZE) (12 elements)
    for lane in listLanes:
        laneQueueTracker.append(traci.lane.getLastStepHaltingNumber(lane))

    # ================ count vehicles in lane
    for lane in listLanes:
        laneNumVehiclesTracker.append(traci.lane.getLastStepVehicleNumber(lane))

    # ================ cum waiting time in minutes
    for lane in listLanes:
        laneWaitingTracker.append(traci.lane.getWaitingTime(str(lane)) / 60)

    # ================ get position matrix of vehicles on lanes
    mapOfCars = getMapOfVehicles(area_length=area_length)

    return [laneQueueTracker, laneNumVehiclesTracker, laneWaitingTracker, mapOfCars]

def get_vehicle_id_entering():
    vehicle_id_entering = []
    entering_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
                     'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']

    for lane in entering_lanes:
        vehicle_id_entering.extend(traci.lane.getLastStepVehicleIDs(lane))

    return vehicle_id_entering

def get_vehicle_id_leaving(vehicle_dict):
    vehicle_id_leaving = []
    vehicle_id_entering = get_vehicle_id_entering()
    for vehicle_id in vehicle_dict.keys():
        if not(vehicle_id in vehicle_id_entering) and vehicle_dict[vehicle_id].entering:
            vehicle_id_leaving.append(vehicle_id)

    return vehicle_id_leaving



def get_car_on_red_and_green(cur_phase):
    listLanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2',
                 'edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
    vehicle_red = []
    vehicle_green = []
    if cur_phase == 1:
        red_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
        green_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
    else:
        red_lanes = ['edge3-0_0', 'edge3-0_1', 'edge3-0_2', 'edge4-0_0', 'edge4-0_1', 'edge4-0_2']
        green_lanes = ['edge1-0_0', 'edge1-0_1', 'edge1-0_2', 'edge2-0_0', 'edge2-0_1', 'edge2-0_2']
    for lane in red_lanes:
        vehicle_red.append(traci.lane.getLastStepVehicleNumber(lane))
    for lane in green_lanes:
        vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        omega = 0
        for vehicle_id in vehicle_ids:
            traci.vehicle.subscribe(vehicle_id, (tc.VAR_DISTANCE, tc.VAR_LANEPOSITION))
            distance = traci.vehicle.getSubscriptionResults(vehicle_id).get(132)
            if distance > 100:
                omega += 1
        vehicle_green.append(omega)

    return max(vehicle_red), max(vehicle_green)

def get_status_img(current_phase,tl_node_id=node_light_7,area_length=600):
    mapOfCars = getMapOfVehicles(area_length=area_length)

    current_observation = [mapOfCars]
    return current_observation

def set_yellow(dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list,node_id="node0"):
    Yellow = "yyyyyyyyyyyyyyyy"
    for i in range(3):
        timestamp = traci.simulation.getCurrentTime() / 1000
        traci.trafficlights.setRedYellowGreenState(node_id, Yellow)
        traci.simulationStep()
        log_rewards(dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list)
        update_vehicles_state(dic_vehicles)

def set_all_red(dic_vehicles,rewards_info_dict,f_log_rewards,rewards_detail_dict_list,node_id="node0"):
    Red = "rrrrrrrrrrrrrrrr"
    for i in range(3):
        timestamp = traci.simulation.getCurrentTime()/1000
        traci.trafficlights.setRedYellowGreenState(node_id, Red)
        traci.simulationStep()
        log_rewards(dic_vehicles, 0, rewards_info_dict, f_log_rewards, timestamp,rewards_detail_dict_list)
        update_vehicles_state(dic_vehicles)

def run(action, current_phase, current_phase_duration, vehicle_dict, rewards_info_dict, f_log_rewards, rewards_detail_dict_list,node_id="node0"):
    return_phase = current_phase
    return_phase_duration = current_phase_duration
    if action == 1:
        set_yellow(vehicle_dict,rewards_info_dict,f_log_rewards, rewards_detail_dict_list,node_id=node_id)
        # set_all_red(vehicle_dict,rewards_info_dict,f_log_rewards, node_id=node_id)
        return_phase, _ = changeTrafficLight_7(current_phase=current_phase)  # change traffic light in SUMO according to actionToPerform
        return_phase_duration = 0
    timestamp = traci.simulation.getCurrentTime() / 1000
    traci.simulationStep()
    log_rewards(vehicle_dict, action, rewards_info_dict, f_log_rewards, timestamp, rewards_detail_dict_list)
    vehicle_dict = update_vehicles_state(vehicle_dict)
    return return_phase, return_phase_duration+1, vehicle_dict



def get_base_min_time(traffic_volumes,min_phase_time):
    traffic_volumes=np.array([36,72,0])
    min_phase_times=np.array([10,35,35])
    for i, min_phase_time in enumerate(min_phase_times):
        ratio=min_phase_time/traffic_volumes[i]
        traffic_volumes_ratio=traffic_volumes/ratio

def phase_vector_to_number(phase_vector,phases_light=phases_light_7):
    phase_vector_7 = []
    result = -1
    for i in range(len(phases_light)):
        phase_vector_7.append(str(get_phase_vector(i)))
    if phase_vector in phase_vector_7:
        return phase_vector_7.index(phase_vector)
    else:
        raise ("Phase vector %s is not in phases_light %s"%(phase_vector,str(phase_vector_7)))



if __name__ == '__main__':
    pass
    print(get_phase_vector(0))
    print(get_phase_vector(1))
    phase_vector_to_number('[0 1 0 1 0 0 1 1 0 1 0 1]')
    pass
    # traci.close()