from CancerRadiationValIter import CancerRadiationValueIter
from CancerRadiationPolIter import CancerRadiationPolIter
from CancerRadiationQLearn import CancerRadiationQLearn
from CancerRadiationMDP import CancerRadiationMDP

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time

class CancerRadiationAnalyzer:

    def __init__(self, MDP, valuePlanner, policyPlanner, Qleaner):
        self.MDP = MDP
        self.planner1 = valuePlanner
        self.planner2 = policyPlanner
        self.Qleaner = Qleaner

    def compare_plannning_methods(self, showProgress = True, includeQlearn = True):
        time1 = time.time()
        V1, pie1, n_iter1 = self.planner1.runValueIteration(showProgress)
        time2 = time.time()
        pie2, n_iter2 = self.planner2.runPolicyInteration(showProgress)
        time3 = time.time()
        if includeQlearn :
            pie3 = self.Qleaner.runQLearning(showProgress)
            time4 = time.time()

        randomized = []
        planned1 = []
        planned2 = []
        planned3 = []

        n = 0
        for i in range(len(pie1)):
            for j in range(len(pie1[i])):
                if pie1[i][j]==None: continue
                n += 1

        print("Number of states ----->  %d\n" %n)
        print("Performance:")
        print("Value iteration  -----> {: >4} interation, {: >.2f} seconds".format(n_iter1, (time2-time1)))
        print("Policy iteration -----> {: >4} interation, {: >.2f} seconds".format(n_iter2, (time3-time2)))
        if includeQlearn :
            print("Q-learning       -----> {: >4} interation, {: >.2f} seconds".format(self.Qleaner.num_episodes, (time4-time3)))
        
        print()

        for i in range(1000):       
            randomized.append(self.MDP.run_with_random_steps(showIterations=False))     
            planned1.append(self.MDP.run_policy(pie1, showIterations=False))
            planned2.append(self.MDP.run_policy(pie2, showIterations=False))
            if includeQlearn :
                planned3.append(self.MDP.run_policy(pie3, showIterations=False))

        ra = np.average(randomized)
        rsd = np.std(randomized)
        p1a = np.average(planned1)
        p1sd = np.std(planned1)
        p2a = np.average(planned2)
        p2sd = np.std(planned2)
        if includeQlearn :
            p3a = np.average(planned3)
            p3sd = np.std(planned3)


        print("Outcome:")
        print("Random planning  -----> %.2f +/- %.2f" %(ra, rsd))
        print("Value iteration  -----> %.2f +/- %.2f" %(p1a, p1sd))
        print("Policy iteration -----> %.2f +/- %.2f" %(p2a, p2sd))
        if includeQlearn :
            print("Q-learning       -----> %.2f +/- %.2f" %(p3a, p3sd))

        sns.kdeplot(randomized, shade=True, label = "random planning")
        sns.kdeplot(planned1, shade=True, label = "value iteration")
        sns.kdeplot(planned2, shade=True, label = "policy iteration")
        if includeQlearn :
            sns.kdeplot(planned3, shade=True, label = "Q-learning")
        plt.show()
    
    # compare two different ways of implementing explorating vs exploitation (annealing vs fixed rate)
    def compare_Qlearn_strategy(self, T0 = 1, epsilon = 0.05, showProgress = True):
        time1 = time.time()
        pieAnneal = self.Qleaner.runQLearning(annealing = True, T0 = T0, showProgress=showProgress)
        time2 = time.time()
        pieEpsilon = self.Qleaner.runQLearning(annealing = False, epsilon = epsilon, showProgress=showProgress)
        time3 = time.time()
        print("Performance:")
        print("Q-leaning with randomized annealing -----> {: >.2f} seconds".format(time2-time1))
        print("Q-leaning with Epsilon exploration ------> {: >.2f} seconds\n".format(time3-time2))
  
        randomized = []
        optimizedAnneal = []
        optimizedEpsilon = []
        for i in range(1000):       
            randomized.append(self.MDP.run_with_random_steps(showIterations=False))     
            optimizedAnneal.append(self.MDP.run_policy(pieAnneal, showIterations=False))
            optimizedEpsilon.append(self.MDP.run_policy(pieEpsilon, showIterations=False))

        ra = np.average(randomized)
        rsd = np.std(randomized)
        oaAnneal = np.average(optimizedAnneal)
        osdAnneal = np.std(optimizedAnneal)
        oaEpsilon = np.average(optimizedEpsilon)
        osdEpsilon = np.std(optimizedEpsilon)

        print("Random planning -----------------------> %.2f +/- %.2f" %(ra, rsd))
        print("Q-leaning with randomized annealing ---> %.2f +/- %.2f" %(oaAnneal, osdAnneal))
        print("Q-leaning with Epsilon exploration ----> %.2f +/- %.2f" %(oaEpsilon, osdEpsilon))

        sns.kdeplot(randomized, shade=True, label = "random planning")
        sns.kdeplot(optimizedAnneal, shade=True, label = "randomized annealing")
        sns.kdeplot(optimizedEpsilon, shade=True, label = "fixed exploration")
        plt.show()   

def main():
    MDP = CancerRadiationMDP(tumor_num=1, OAR_num=3, tumor_size=8, OAR_size=5)
    planner1 = CancerRadiationValueIter(MDP, 0.9, 0.0001, 100)   
    planner2 = CancerRadiationPolIter(MDP, 0.9, 0.01, 50)
    planner3 = CancerRadiationQLearn(MDP, 0.9, 0.1, 10000)
    analyzer = CancerRadiationAnalyzer(MDP, planner1, planner2, planner3)
    analyzer.compare_plannning_methods(includeQlearn = False)
    analyzer.compare_Qlearn_strategy()

if __name__ == "__main__":
     main()   