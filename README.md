# CancerRadiationMDP
A Markov Decision Process (MDP) with large number of states and its solvers (value iteration, policy iteration and Q-learning)

## Introduction
This problem is partially inspired by [Kim M et al.](https://www.ncbi.nlm.nih.gov/pubmed/19556687) When doing radiation therapy to treat cancers, the outcome is evaluated by the eradication of tumor(s) and the damage to the surrounding tissues, also known as **Organs at Risk (OARs)**. OARs are more resistant to radiation and an optimal treatment plan needs to ensure tumor eradication and minimal OAR damage. How to choose different doses of radiation at different stages of treatment in response to different levels of tumor shrinkage and tissue damage is key to the success of radiation therapy.

In our model, the patient can have 1~9 tumors and 1~9 OARs.  Each tumor and OAR can have a size of 0~9. The sizes of each tumor and OAR define the states of the MDP. The radiation can have 3 levels by default, 0 (low), 1 (medium) and 2 (high) which defines the action space. The transition function T is defined by the probability of a tumor or OAR to remain the same or reduce size by 1 during a treatment session (a step) under different levels of radiations. There is a negative reward when an OAR reaches 0 size (tissue loss) and a positive reward when a tumor reaches 0 size (tumor removal). This model is illustrated in the figure below.

