from CancerRadiationMDP import CancerRadiationMDP
from CancerRadiationUtility import CancerRadiationUtility
import copy
import progressbar

class CancerRadiationValueIter:

    def __init__(self, MDP, gamma, threshold, max_iter):
        self.MDP = MDP
        self.utility = CancerRadiationUtility(MDP, gamma)
        self.threshold = threshold
        self.max_iter = max_iter
        self.Vmatrix = []
        self.pie = []
        self.num_iter = 0

    # Value interation
    def __getVmatrix(self, showProgress):
        V = self.utility.initializMatrix(0)
        n = 0       
        if showProgress:
            print("Value Iteration in progress:")
            bar = progressbar.ProgressBar(maxval=self.max_iter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        while True:
            n += 1
            if showProgress: bar.update(n)
            V_hat = self.utility.initializMatrix(0)
            # select one state and calulate its V hat, iterate through whole value matirx
            # it makes sure that only legit states are selected and sorted in order so there is no repeat states
            for T in range(1, len(V)):                                        # T = 0 is not necessary as it is the exit condition
                Tlist = self.MDP.stateNumToList(T, self.MDP.numTumors)
                if max(Tlist)>self.MDP.initialTumorSize: continue               # skip empty cells
                TlistSorted = sorted(Tlist)
                if Tlist != TlistSorted: continue                               # skip repeated cells
                for OAR in range(1, len(V[T])):                               # OAR = 0 is also not necessary
                    OARlist = self.MDP.stateNumToList(OAR, self.MDP.numOARs)
                    if max(OARlist)>self.MDP.initialOARSize: continue           # skip empty cells
                    OARlistSorted = sorted(OARlist)
                    if OARlist != OARlistSorted: continue                       # skip repeated cells                                                
                    value_max = -1000
                    for action in range(self.MDP.numAction):
                        value = self.utility.caluculateVhat(Tlist, OARlist, action, V)
                        if value_max < value: value_max = value
                    V_hat[T][OAR] = value_max

            if n > self.max_iter:
                Warning.warn("value interation did not converge")
                break                
            if self.utility.check_diff(V, V_hat)<self.threshold: break
            V = copy.copy(V_hat)
        if showProgress: bar.finish()
        self.Vmatrix = V
        self.num_iter = n

    def runValueIteration(self, showProgress = False):
        self.__getVmatrix(showProgress)
        self.pie = self.utility.getPolicy(self.Vmatrix)
        return self.Vmatrix, self.pie, self.num_iter