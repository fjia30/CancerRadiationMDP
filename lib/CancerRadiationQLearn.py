from CancerRadiationUtility import CancerRadiationUtility
from CancerRadiationMDP import CancerRadiationMDP
import numpy as np
import progressbar
import random
import copy

class CancerRadiationQLearn:
    def __init__(self, MDP, gamma, alpha, num_episodes):
        self.MDP = MDP
        self.utility = CancerRadiationUtility(MDP, gamma)
        self.gamma = gamma
        self.alpha = alpha
        self.num_episodes = num_episodes
        self.Qmap = {}
        self.pie = []

    def __initializeQmap(self):
        V = self.utility.initializMatrix(None)
        Qmap = {}
        for T in range(len(V)):
            Tlist = self.MDP.stateNumToList(T, self.MDP.numTumors)
            if max(Tlist)>self.MDP.initialTumorSize: continue
            TlistSorted = sorted(Tlist)
            if Tlist != TlistSorted: continue
            for OAR in range(len(V[T])):
                OARlist = self.MDP.stateNumToList(OAR, self.MDP.numOARs)
                if max(OARlist)>self.MDP.initialOARSize: continue
                OARlistSorted = sorted(OARlist)
                if OARlist != OARlistSorted: continue
                Qmap[(T, OAR)] = [0]*self.MDP.numAction
        return Qmap

    def __QLearning(self, annealing, T0, epsilon, showProgress):
        Qmap = self.__initializeQmap()      
        if showProgress:
            print("Q-learning in progress:")
            bar = progressbar.ProgressBar(maxval=self.num_episodes, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        for i in range(self.num_episodes):
            if showProgress: bar.update(i)
            OARstate, Tstate = self.MDP.reset()
            done = False
            while not done:
                if annealing: action = np.argmax(Qmap[(Tstate, OARstate)] + np.random.randn(1,self.MDP.numAction)*(T0*1.0/(i+1)))
                elif random.random() < epsilon: action = random.randrange(self.MDP.numAction)
                else: action = np.argmax(Qmap[(Tstate, OARstate)])
                OARstateNew, TstateNew, reward, done = self.MDP.step(action)
                Qmap[(Tstate, OARstate)][action] = (1-self.alpha)* Qmap[(Tstate, OARstate)][action] + self.alpha*(reward + self.gamma*np.max(Qmap[(TstateNew, OARstateNew)]))
                OARstate = OARstateNew
                Tstate = TstateNew

        if showProgress: bar.finish()
        self.Qmap = Qmap


    def __getPolicy(self):
        pie = self.utility.initializMatrix(None)
        for key in self.Qmap.keys():
            pie[key[0]][key[1]] = np.argmax(self.Qmap[key])
        self.pie = pie

    # stratagy can be either RA: randomized anealing or Epsilon: a fixed rate of exploration
    def runQLearning(self, annealing = False, T0 = 1, epsilon = 0.05, showProgress = False):
        self.__QLearning(annealing, T0, epsilon, showProgress)
        self.__getPolicy()
        return self.pie