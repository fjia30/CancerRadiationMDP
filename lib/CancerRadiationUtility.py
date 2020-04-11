import copy

class CancerRadiationUtility:

    def __init__(self, MDP, gamma):
        self.MDP = MDP
        self.gamma = gamma
    
    def initializMatrix(self, value):
        self.MDP.reset()
        OARstates = self.MDP.OARstate+1
        Tstates = self.MDP.Tstate+1
        V = [[value]*OARstates for _ in range(Tstates)]
        return V

    # Check convergence
    def check_diff(self, V1, V2):
        max_diff = -1
        for i in range(len(V1)):
            for j in range(len(V1[i])):
                diff = abs(V1[i][j]-V2[i][j])
                if max_diff < diff: max_diff = diff
        return max_diff

   # each member of the probMap is state : probability of that state
    def __caluculateProbMap(self, statelist, reduceProb):
        sameProb = 1 - reduceProb
        probMatrix = []                             # each member of the probMatirx is [ state[], probability of that state]   
        # Scan state list
        # at each position in state list, can be no change or -1    
        for state in statelist:
            if state == 0 : continue
            if probMatrix == []: 
                probMatrix.append([[state], sameProb])
                probMatrix.append([[state-1], reduceProb])
                continue
            newList = copy.copy(probMatrix)
            probMatrix = []      
            for i in newList:                       # eg. i is [[OAR1, OAR2], probability], now we add OAR3 to it
                newSameState = i[0] + [state]
                newSameProb = i[1] * sameProb
                newReducedState = i[0] + [state-1]
                newReducedProb = i[1] * reduceProb
                probMatrix.append([newSameState, newSameProb])
                probMatrix.append([newReducedState, newReducedProb])       
        # create probMap from probMatrix
        probMap = {}
        for i in probMatrix:
            state = self.MDP.stateListToNum(i[0])
            if state in probMap: probMap[state] += i[1]
            else: probMap[state] = i[1]

        return probMap

    # calulate a Vhat value from the state and action
    # Vhat(s) = max(a) { sum { T(s,a,s') * [R(s,a,s') + gamma * V(s')] } }
    # this function caluculate sum { T(s,a,s') * [R(s,a,s') + gamma * V(s')] } for a given action
    #                                    p            r            f
    def caluculateVhat(self, Tlist, OARlist, action, Vmatrix):
        OARreduceProb = self.MDP.transitionRate[action][0]
        TreduceProb = self.MDP.transitionRate[action][1]
        OARproMap = self.__caluculateProbMap(OARlist, OARreduceProb)
        TproMap = self.__caluculateProbMap(Tlist, TreduceProb)
        value = 0
        for OAR in OARproMap:
            for T in TproMap:
                p = OARproMap[OAR] * TproMap[T]
                r = self.MDP.calculateReward(OARlist, self.MDP.stateNumToList(OAR, self.MDP.numOARs), 
                                             Tlist, self.MDP.stateNumToList(T, self.MDP.numTumors))
                f = self.gamma * Vmatrix[T][OAR]
                value += p * (r + f)
        return value
    
    # given Vmatrix, calculate the optimal policy
    def getPolicy(self, Vmatrix):
        pie = self.initializMatrix(None)
        for T in range(1, len(Vmatrix)):                               # T = 0 is not necessary as it is the exit condition
            Tlist = self.MDP.stateNumToList(T, self.MDP.numTumors)
            if max(Tlist)>self.MDP.initialTumorSize: continue               # skip empty cells
            TlistSorted = sorted(Tlist)
            if Tlist != TlistSorted: continue                               # skip repeated cells

            for OAR in range(1, len(Vmatrix[T])):                           # OAR = 0 is also not necessary
                OARlist = self.MDP.stateNumToList(OAR, self.MDP.numOARs)
                if max(OARlist)>self.MDP.initialOARSize: continue           # skip empty cells
                OARlistSorted = sorted(OARlist)
                if OARlist != OARlistSorted: continue                       # skip repeated cells                                                
                value_max = -1000
                optimalAction = None
                for action in range(self.MDP.numAction):
                    value = self.caluculateVhat(Tlist, OARlist, action, Vmatrix)
                    if value_max < value: 
                        value_max = value
                        optimalAction = action
                pie[T][OAR] = optimalAction
        return pie