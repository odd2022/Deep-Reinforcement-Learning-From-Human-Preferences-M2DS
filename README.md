# Deep-Reinforcement-Learning-From-Human-Preferences-M2DS

This repo shows our work on the article[Deep Reinforcement Learning from Human
Preferences](https://blog.openai.com/deep-reinforcement-learning-from-human-preferences/),
based on the paper <https://arxiv.org/pdf/1706.03741.pdf>.

To structure our code, we inspired ourselves from the pages [Deep Reinforcement Learning from Human Preferences](https://github.com/mrahtz/learning-from-human-preferences) and [Deep Reinforcement Learning from Human Preferences](https://github.com/HumanCompatibleAI/learning-from-human-preferences). 
However, we created a simpler structure, compatible with a recent version of python and torch (the work done in this two github repos is done in python 3.7 and tensorflow)

We focused on the Atari Game "Pong". 
We use python 3.9 in this repo

# Organisation of the folders 

## Pong_PPO_DQN 

This repository was created to test different classic RL algorithms on the Pong Game to discover the game and different algortihms. 

- pong_DQN_bhctsntrk.py : we tested the code provided by [DQN Algorithm for Solving Atari Pong](https://github.com/bhctsntrk/OpenAIPong-DQN) to have an idea of the results we should have. 

- pong_DQN_train.py and pong DQN_test.py: test of the DQN algorithm 

- pong_PPO_train.py, pong PPO_test.py, pong_PPO_train_improved.py: test of the PPO algorithm 


## Pong_A2C_classic

## Pong_human_pref

In this section me implemented the algorithm with human preferences

- drlhp_test_A2C_video_pretrain.py we pretrain the reward model before using the algo + video rendering for preferences 

- drlhp_test_A2C_video.py and drlhp_train_A2C_video.py we train the model with human preferences and we render video to be able to chose correctly the preferences. 

- drlhp_test_A2C.py and drlhp_train_A2C.py: we took this model as a basis but here you don't have video rendering so you can't really chose between the preferences. 

