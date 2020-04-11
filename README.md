# CancerRadiationMDP
A Markov Decision Process (MDP) with large number of states and its solvers (value iteration, policy iteration and Q-learning)

## Introduction
This problem is partially inspired by [Kim M et al.](https://www.ncbi.nlm.nih.gov/pubmed/19556687) When doing radiation therapy to treat cancers, the outcome is evaluated by the eradication of tumor(s) and the damage to the surrounding tissues, also known as **Organs at Risk (OARs)**. OARs are more resistant to radiation and an optimal treatment plan needs to ensure tumor eradication and minimal OAR damage. How to choose different doses of radiation at different stages of treatment in response to different levels of tumor shrinkage and tissue damage is key to the success of radiation therapy.

In our model, the patient can have 1-9 tumors and 1-9 OARs.  Each tumor and OAR can have a size of 0-9. The sizes of each tumor and OAR define the states of the MDP. A problem with 1 tumor of size 6 and 3 OARs of size 4 has ~200 states. A problem with 2 tumors of size 8 and 5 OARs of size 5 has ~11,000 unique states.

The radiation can have 3 levels by default, 0 (low), 1 (medium) and 2 (high) which defines the action space. 

The transition function T is defined by the probability of a tumor or OAR to remain the same or reduce size by 1 during a treatment session (a step) under different levels of radiations. There is a negative reward when an OAR reaches 0 size (tissue loss) and a positive reward when a tumor reaches 0 size (tumor removal). This model is illustrated in the figure below.

![CancerRadiationMDP](https://github.com/fjia30/CancerRadiationMDP/blob/master/CancarRadiationMDP.png)

## Package content
class name | description
---------------- | ------------------
[CancerRadiationMDP.py](https://github.com/fjia30/CancerRadiationMDP/blob/master/lib/CancerRadiationMDP.py) | Simulates the MDP problem
[CancerRadiationUtility.py](https://github.com/fjia30/CancerRadiationMDP/blob/master/lib/CancerRadiationUtility.py) | A utility class used by the others
[CancerRadiationValIter.py](https://github.com/fjia30/CancerRadiationMDP/blob/master/lib/CancerRadiationValIter.py) | A solver of the MDP using value iteration
[CancerRadiationPolIter.py](https://github.com/fjia30/CancerRadiationMDP/blob/master/lib/CancerRadiationPolIter.py) | A solver of the MDP using policy iteration
[CancerRadiationQLearn.py](https://github.com/fjia30/CancerRadiationMDP/blob/master/lib/CancerRadiationQLearn.py) | A solver of the MDP using Q-learning
[CancerRadiationAnalyzer.py](https://github.com/fjia30/CancerRadiationMDP/blob/master/lib/CancerRadiationAnalyzer.py) | Analyze the and compares different solvers

## Note
This model can be extended to other problems. In general, we define some targets (tumors in our case) and some off-targets (OARs in our case). The "sizes" of the targets and off-targets define their progress. The transition probabilities define the probabilities to progress during each step for targets and off-targets. There is a positive reward to remove each target (size bacomes 0) and a negative reward to remove each off-target. It is also possible to add a small penality or price at each step.