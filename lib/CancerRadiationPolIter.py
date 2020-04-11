from CancerRadiationMDP import CancerRadiationMDP
from CancerRadiationValIter import CancerRadiationValueIter
from CancerRadiationUtility import CancerRadiationUtility
import progressbar
import copy

class CancerRadiationPolIter(CancerRadiationValueIter):

    def __init__(self, MDP, gamma, threshold, max_iter):
        self.MDP = MDP
        self.utility = CancerRadiationUtility(MDP, gamma)
        self.threshold = threshold
        self.max_iter = max_iter
        self.pie = []
        self.num_iter = 0

    def runPolicyInteration(self, showProgress = False):
        # initialize value and policy
        V = self.utility.initializMatrix(0)
        pie = self.utility.initializMatrix(0)
        n = 0
        if showProgress:
            print("Policy Iteration in progress:")
            bar = progressbar.ProgressBar(maxval=self.max_iter, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
        while True:
            n += 1
            if showProgress: bar.update(n)
            # get value matrix from policy
            while True:
                V_hat = self.utility.initializMatrix(0)
                for T in range(1, len(V)):
                    Tlist = self.MDP.stateNumToList(T, self.MDP.numTumors)
                    if max(Tlist)>self.MDP.initialTumorSize: continue
                    TlistSorted = sorted(Tlist)
                    if Tlist != TlistSorted: continue
                    for OAR in range(1, len(V[T])):
                        OARlist = self.MDP.stateNumToList(OAR, self.MDP.numOARs)
                        if max(OARlist)>self.MDP.initialOARSize: continue
                        OARlistSorted = sorted(OARlist)
                        if OARlist != OARlistSorted: continue    
                        # for every reasonable T and OAR state, get the action according to the policy                                      
                        action = pie[T][OAR]
                        V_hat[T][OAR] = self.utility.caluculateVhat(Tlist, OARlist, action, V)
                if self.utility.check_diff(V, V_hat)<self.threshold: break
                V = copy.copy(V_hat)
            
            # get policy from values
            pie_hat = self.utility.getPolicy(V)
            if pie==pie_hat: break
            pie = copy.copy(pie_hat)
            if n>self.max_iter:
                Warning.warn("Policy interation did not converge")
                break
        if showProgress: bar.finish()
        self.pie = pie
        self.num_iter = n

        return self.pie, self.num_iter

