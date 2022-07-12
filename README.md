# AAS_Project
 **Autonomous and Adaptive System - Project**

 Implementation of the Deep Deterministic Policy Gradient [1], and Soft Actor Critic [2]

## References

[1] Timothy P. Lillicrap et al. Continuous control with deep reinforcement learning. 2015. DOI:
10.48550/ARXIV.1509.02971. URL: https://arxiv.org/abs/1509.02971.

[2] Tuomas Haarnoja et al. Soft Actor-Critic Algorithms and Applications. 2018. DOI: 10.48550/
ARXIV.1812.05905. URL: https://arxiv.org/abs/1812.05905.

## Commands

To run an algorithm there must be a json file in the config directory with the required hyperparameters, named after the algorithm `*name*.json` with `*name*` in (`ddpg*`, `ddpg_dc*` or `sac*`)
 
To train an algorithm

```
python train.py [-r] *name*
optional arguments:
   -r, --render        render the environment
```

To test the best weigths of an algorithm:
```
python test.py [-r] *name*
optional arguments:
   -r, --render        render the environment
```

To test the whole evolution of an algorithm:
```
python evolution.py [-r] *name*
optional arguments:
   -r, --render        render the environment
```