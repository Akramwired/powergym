# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

"""Random agent to probe enviroment
"""
import matplotlib.pyplot as plt
import numpy as np
import imageio
import glob
from stable_baselines3 import PPO
from powergym.env_register import make_env, remove_parallel_dss

import argparse
import random
import itertools
import sys, os
import multiprocessing as mp


models_dir = "models/PPO"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)



cwd = os.getcwd()
env = make_env('9Bus')
env.seed(123456 + 0)
env.reset()

print('This system has {} capacitors, {} regulators and {} batteries'.format(env.cap_num, env.reg_num,
                                                                             env.bat_num))
print('reg, bat action nums: ', env.reg_act_num, env.bat_act_num)
print('-' * 80)

if not os.path.exists(os.path.join(cwd, 'random_agent_plots')):
    os.makedirs(os.path.join(cwd, 'random_agent_plots'))

model = PPO('MlpPolicy',
                env=env,
                seed=0,
                batch_size=64,
                ent_coef=0.01,
                learning_rate=0.0003,
                n_epochs=10,
                n_steps=64, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000

for i in range(30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{models_dir}/{TIMESTEPS*i}")
