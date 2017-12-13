
# coding: utf-8

# In[ ]:

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from enum import Enum
import numpy as np
import pandas as pd
import os
import random
import math

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_SELECT_IDLE_WORKER = actions.FUNCTIONS.select_idle_worker.id
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id
_HARVEST_RETURN = actions.FUNCTIONS.Harvest_Return_quick.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index
_PLAYER_SELF = 1
_IDLE_WORKER_COUNT = 7


_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45 
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_MINERAL = 341
_SCREEN = [0]
_NOT_QUEUED = [0]
_QUEUED = [1]


ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_SCV = 'selectscv'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_SELECT_BARRACKS = 'selectbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_SELECT_BARRACKS,
    ACTION_BUILD_MARINE,
]

ARMY_REWARD = 0.2

class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions)

    def choose_action(self, observation):
        self.check_state_exist(observation)
        
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.ix[observation, :]
            
            state_action = state_action.reindex(np.random.permutation(state_action.index))
            
            action = state_action.values.argmax()
        else:
            action = np.random.choice(self.actions)
            
        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)
        
        q_predict = self.q_table.ix[s, a]
        q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))
            
            
class Buildings:
    amount = 0
    order = 0
    width = 0
    screen_width = 84
    positions = []
    checked_position = [-1,-1]
    array =  []
      
    def __init__(self,name,width):
        self.name = name
        self.amount = 0
        self.order = 0
        self.width = width
        self.positions = []
    
    def add(self):
        self.positions.append(self.checked_position)
        self.amount = self.amount+1
        self.build()
        
    def reset(self):
        self.amount = 0
        self.order = 0
        self.positions = []
        self.checked_position = [-1,-1]
        
    def firstAreaChnage(self,obs):
        Buildings.array =  np.zeros((self.screen_width,self.screen_width))
        Buildings.array.dtype = int
        unit_type = obs.observation["screen"][_UNIT_TYPE]
        for i in range(0,self.screen_width):
            for j in range(0,self.screen_width):
                Buildings.array[i][j] = unit_type[i][j]
        
        mineral_y, mineral_x = (Buildings.array==_MINERAL).nonzero()
        command_y, command_x = (Buildings.array==_TERRAN_COMMANDCENTER).nonzero()
        x_min = mineral_x.min()
        if command_x.min()<x_min:
            x_min = command_x.min()
        x_max = mineral_x.max()
        if command_x.max()>x_max:
            x_max = command_x.max()
        y_min = mineral_y.min()
        if command_y.min()<y_min:
            y_min = command_y.min()
        y_max = mineral_y.max()
        if command_y.max()>y_max:
            y_max = command_y.max()
        boundary = []
        boundary_size=0
        x_gap = x_max-x_min
        y_gap = y_max-y_min
        for i in range(y_min,y_max+1):
            for j in range(x_min,x_max+1):
                Buildings.array[i][j] = 1
                if i==y_min or i==y_max or j==x_min or j==x_max: 
                    boundary.append([j,i])
                    boundary_size = boundary_size+1
        Buildings.array = self.boundaryExpand(boundary, boundary_size)
        return Buildings.array
    
    def check(self):
        half = (int)((self.width-1)/2)
        stack = self.width*self.width
        find_value = 2+half 
        check = 0
        position = self.randomPosition(find_value)
        Buildings.array[position[1]][position[0]] = 0
        if position!=[-1,-1]:
            for i in range(0,self.width):
                for j in range(0, self.width):
                    y = position[1]+half-i
                    x = position[0]+half+j
                    if y>0 and y<self.screen_width and x>0 and x<self.screen_width:
                        if Buildings.array[y][x]!=1:
                            check=check+1
            if check != stack:
                position = self.check()
            self.checked_position = position
        return position
    
    def build(self):
        half = (int)((self.width-1)/2)
        stack = self.width*self.width
        find_value = 2+half 
        boundary = []
        boundary_size=0
        position = self.checked_position
        if position!=[-1,-1]:
            for i in range(0,self.width):
                for j in range(0, self.width):
                    y = position[1]+half-i
                    x = position[0]+half+j
                    if y>0 and y<self.screen_width and x>0 and x<self.screen_width:
                        Buildings.array[y][x] = 1
                        if i==0 or i==self.width-1 or j==0 or j == self.width-1:
                            boundary.append([x,y])
                            boundary_size = boundary_size+1
            Buildings.array = self.boundaryExpand(boundary, boundary_size)
            self.checked_position =[-1,-1]

    def boundaryExpand(self, boundary, boundary_size):
        expand_range = 6

        for i in range(0,boundary_size):
            for j in range(-1,2):
                for k in range(-1,2):
                    if not(j==0 and k==0):
                        for l in range(0, expand_range):
                            x = boundary[i][0]+k*(l+1)
                            y = boundary[i][1]+j*(l+1)
                            if x>=0 and y>=0 and x<self.screen_width and y<self.screen_width:
                                if Buildings.array[y][x] !=1 and (Buildings.array[y][x]==0 or Buildings.array[y][x]>(l+2)):
                                    Buildings.array[y][x] = (l+2)
        return Buildings.array

    def randomPosition(self,find_value):
        find_y, find_x = (self.array==find_value).nonzero()
        if find_y.any():
            rand = random.randint(0, find_x.size-1)
            rand_x = find_x[rand]
            rand_y = find_y[rand]
            return [rand_x,rand_y]
        else:
            return[-1,-1]

class Scv_Actions(Enum):
    nothing = 0
    build_supply= 1
    build_barrack = 2

class FinalAgent(base_agent.BaseAgent):
    
    scv_selected = False
    supplys = Buildings("supply_depot",7)
    barracks = Buildings("barrack",11)
    next_action = Scv_Actions.nothing
    first_check = False
    
    previous_episode = 0
    learning_per_episode = 0
    mineral_check = 0
    
    def __init__(self):
        super(FinalAgent, self).__init__()
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.previous_army_supply = 0
        self.previous_action = None
        self.previous_state = None
        
    def step(self, obs):
        super(FinalAgent, self).step(obs)
        
        if self.episodes>self.previous_episode:
            if obs.observation["player"][_IDLE_WORKER_COUNT]>0:
                if (_HARVEST_RETURN in obs.observation['available_actions'] and _HARVEST_GATHER in obs.observation['available_actions']) or _HARVEST_GATHER not in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_IDLE_WORKER,[_NOT_QUEUED])
                elif _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_type = obs.observation["screen"][_UNIT_TYPE]
                    unit_y, unit_x = (unit_type == _MINERAL).nonzero()
                    if unit_y.any():
                        self.mineral_check = self.mineral_check+1
                        i = self.mineral_check
                        if i>len(unit_y)-1:
                            self.mineral_check = 0
                            i = 0
                        target = [unit_x[i], unit_y[i]]
                        return actions.FunctionCall(_HARVEST_GATHER, [_SCREEN, target])
            else:
                self.scv_selected = False
                self.supplys.reset()
                self.barracks.reset()
                self.supplys.firstAreaChnage(obs)
                self.next_action = Scv_Actions.nothing
                
                print("episode : ",self.previous_episode, "reward : ", self.reward,"learning_per_episode : " , self.learning_per_episode)
                self.learning_per_episode = 0
                self.previous_episode = self.episodes
                self.reward = 0
        
        else:
            supply_limit = obs.observation['player'][4]
            army_supply = obs.observation['player'][5]
            current_state = [
                self.supplys.amount,
                self.barracks.amount,
                supply_limit
            ]
            smart_action = ACTION_DO_NOTHING
            if self.next_action == Scv_Actions.nothing:
                if self.previous_action is not None:
                    reward = 0

                    if army_supply > self.previous_army_supply:
                        value = army_supply - self.previous_army_supply
                        value = value/5
                        if value>1:
                            value = 1
                        reward += value

                    self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

                rl_action = self.qlearn.choose_action(str(current_state))
                smart_action = smart_actions[rl_action]
                self.previous_army_supply = army_supply
                self.previous_state = current_state
                self.previous_action = rl_action
                
                self.learning_per_episode = self.learning_per_episode+1
            
            else:
                if obs.observation["player"][_IDLE_WORKER_COUNT]==0:
                    if self.next_action == Scv_Actions.build_supply:
                        self.next_action = Scv_Actions.nothing
                        self.supplys.add()

                    elif self.next_action == Scv_Actions.build_barrack:
                        self.next_action = Scv_Actions.nothing
                        self.barracks.add()
                
                if self.next_action == Scv_Actions.build_supply:
                    smart_action = ACTION_BUILD_SUPPLY_DEPOT
                    
                elif self.next_action == Scv_Actions.build_barrack:
                    smart_action = ACTION_BUILD_BARRACKS
            
            if smart_action == ACTION_DO_NOTHING:
                return actions.FunctionCall(_NO_OP, [])
            
            elif smart_action == ACTION_SELECT_SCV:
                if self.scv_selected==False:
                    if _HARVEST_GATHER not in obs.observation['available_actions']: 
                        unit_type = obs.observation['screen'][_UNIT_TYPE]
                        unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()
                        if unit_y.any():
                            i = random.randint(0, len(unit_y) - 1)
                            target = [unit_x[i], unit_y[i]]
                            return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
                    else:
                        self.scv_selected = True


                elif obs.observation["player"][_IDLE_WORKER_COUNT]==1:
                    return actions.FunctionCall(_SELECT_IDLE_WORKER,[_NOT_QUEUED])

            elif smart_action == ACTION_BUILD_SUPPLY_DEPOT and (obs.observation["player"][_IDLE_WORKER_COUNT]>0 or self.supplys.amount==0):
                if _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    target = self.supplys.check()
                    if target!=[-1,-1]:
                        self.next_action = Scv_Actions.build_supply
                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_SCREEN, target])
                    else:
                        self.next_action = Scv_Actions.nothing

            elif smart_action == ACTION_BUILD_BARRACKS and obs.observation["player"][_IDLE_WORKER_COUNT]>0:
                if _BUILD_BARRACKS in obs.observation['available_actions']:
                    target = self.barracks.check()
                    if target!=[-1,-1]:
                        self.next_action = Scv_Actions.build_barrack

                        return actions.FunctionCall(_BUILD_BARRACKS, [_SCREEN, target])
                    else:
                        self.next_action = Scv_Actions.nothing

            elif smart_action == ACTION_SELECT_BARRACKS:
                if(self.barracks.amount>0):
                    self.barracks.order=self.barracks.order+1
                    if self.barracks.order>(self.barracks.amount-1):
                        self.barracks.order = 0
                    target = self.barracks.positions[self.barracks.order]
                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            
        
        return actions.FunctionCall(_NO_OP, [])

