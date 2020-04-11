#!/usr/bin/env python
# coding: utf-8

##################################
# for each tumor and organ at risk (OAR), the state is 0~4, 0 being none-exsiting, 4 being original size
# for an OAR, 0 means loss of organ and a reward of -1 is awarded
# for a tumor, 0 means eradication and a positive reward is awarded. When all tumors reach 0, treatment is finished
# the positive reward for a tumor is calulated so that losing all OARs and all tumors has a net reward of 0
# states [tumorState, OARstate] OARstate is a number 0~4444 with smaller digits in the front (123, 1344, 2233, etc.)
# actions are 0: low radiation, 1 mid radiation, 2 high radiation
##################################

import random
import copy

class CancerRadiationMDP:

    def __init__(self, tumor_num=1, OAR_num=5, tumor_size=5, OAR_size=5, step_reward=0, action_num=3, tr=None):

        self.numTumors = tumor_num      # number of tumors
        self.numOARs = OAR_num          # number of OARs
        self.numAction = action_num     # default transition rates for OAR and tumors under different radiation conditions
                                        

        

        if action_num!=3 and tr==None: raise ValueError("transition probability must be specified if action space is not 3")
        if tr==None:                                # transition probability, 0~1, the probability of OAR or tumor to have size - 1 during a treatmene session
            self.transitionRate =  [[0.2, 0.4],     # transition probability for action 0 [OAR, Tumor]
                                    [0.4, 0.6],     # transition probability for action 1 [OAR, Tumor]
                                    [0.5, 0.8]]     # transition probability for action 2 [OAR, Tumor]     
        elif len(tr)!=action_num: raise ValueError("transition probability matrix must have the same length as the action space")
        elif min([len(i) for i in tr])<2: raise ValueError("transition probability matrix must have at least 2 columns and only the first 2 colums will be used")
        else: self.transitionRate = copy.copy(tr)

        # rewards[0]: reward for each OAR becoming 0 (tissue loss), fixed at -1
        # rewards[1]: reward for each tumor becoming 0 (eradication)
        # reword is calculated so that if all tumors and OARs become 0 the total reward is 0
        # rewards[2]: step reward, usually a penalty for each step during radiation treatment to encourage shorter treatment regimen
        OARreward = -1
        Treward = (OARreward * OAR_num * -1) / tumor_num
        self.rewards = [OARreward, Treward, step_reward]

        if tumor_size>0 and tumor_size<10: self.initialTumorSize = tumor_size 
        else: raise ValueError("tumor_size needs to be >0 and <10")

        if OAR_size>0 and OAR_size<10: self.initialOARSize = OAR_size 
        else: raise ValueError("OAR_size needs to be >0 and <10")

        self.OARstate = None
        self.Tstate = None
        self.totalReward = None

    def stateNumToList(self, state, length):

        return [int(i) for i in list("{0:0>{1}d}".format(state, length))]

    def stateListToNum(self, state):
        state.sort()   
        return int("".join([str(i) for i in state]))


    def radiation(self, state, probability):
        if random.random()<probability and state > 0:
            state -= 1
        return state
    
    def __calculateStateDiff(self, SList, SprimeList):
        zerosInListS = 0
        zerosInListSprime = 0
        for i in range(len(SList)): 
            if SList[i]==0: zerosInListS += 1
            if SprimeList[i]==0: zerosInListSprime += 1
        return zerosInListSprime - zerosInListS

    
    # calulate the immediate reward
    # reward is dependent on s and s_prime as going from 111 to 001 gives 2*reward but frm 011 ti 001 only gives 1*reward
    def calculateReward(self, OARSList, OARSprimeList, TSList, TSprimeList):
        OARdiff = self.__calculateStateDiff(OARSList, OARSprimeList)
        Tdiff = self.__calculateStateDiff(TSList, TSprimeList)
        reward = OARdiff * self.rewards[0] + Tdiff * self.rewards[1]
        if reward == 0: reward = self.rewards[2]
        return reward

    def reset(self):
        self.OARstate = self.stateListToNum([self.initialOARSize]*self.numOARs)
        self.Tstate = self.stateListToNum([self.initialTumorSize]*self.numTumors)
        self.totalReward = 0
        return self.OARstate, self.Tstate
    
    # print out the current state for OARs and tumors
    # initialize the MDP if necessary
    def render(self, action):
        if self.OARstate == None or self.Tstate == None or self.totalReward == None: self.reset()
        print("Tumors: ", end="")
        [print("{: >5d}".format(i), end="") for i in self.stateNumToList(self.Tstate, self.numTumors)]
        print("    OARs: ", end="")
        [print("{: >5d}".format(i), end="") for i in self.stateNumToList(self.OARstate, self.numOARs)]
        print("    Reward: {: >.4f}".format(self.totalReward), end="")
        if action != None: print("    Action: {: >4d}".format(action))
        else: print()

    # take a step (doing one treatment session) with a given action (radiation dosage)
    def step(self, action):
        if self.OARstate == None or self.Tstate == None or self.totalReward == None: self.reset()
        isFinished = False        
        OARstate = self.OARstate
        Tstate = self.Tstate

        OARstate = self.stateNumToList(OARstate, self.numOARs)
        OARstateNew = []
        for state in OARstate:
            newState = self.radiation(state, self.transitionRate[action][0])
            OARstateNew.append(newState)
        

        Tstate = self.stateNumToList(Tstate, self.numTumors)
        TstateNew = []
        for state in Tstate:
            newState = self.radiation(state, self.transitionRate[action][1])
            TstateNew.append(newState)

        reward = self.calculateReward(OARstate, OARstateNew, Tstate, TstateNew) 
        OARstateNew = self.stateListToNum(OARstateNew)     
        TstateNew = self.stateListToNum(TstateNew)
         
        if OARstateNew == 0 or TstateNew == 0: isFinished = True
        self.OARstate = OARstateNew
        self.Tstate = TstateNew
        self.totalReward += reward
        return OARstateNew, TstateNew, reward, isFinished


    def run_with_random_steps(self, showIterations=True):
        done = False
        while not done:
            action = random.randint(0,self.numAction-1)
            if showIterations: self.render(action)
            OARstateNew, TstateNew, reward, done = self.step(action)
        if showIterations: self.render(None)
        return self.totalReward

    # run the whole MDP with a given policy pie
    def run_policy(self, pie, showIterations=True):
        OARstate, Tstate = self.reset()
        done = False        
        while not done:
            action = pie[Tstate][OARstate]
            if showIterations: self.render(action)
            OARstate, Tstate, reward, done = self.step(action)      
        if showIterations: self.render(None)
        return self.totalReward