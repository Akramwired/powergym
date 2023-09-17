# Copyright 2021 Siemens Corporation
# SPDX-License-Identifier: MIT

"""Random agent to probe enviroment
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
cwd = os.getcwd()
if not os.path.exists(os.path.join(cwd,'random_agent_plots')):
    os.makedirs(os.path.join(cwd,'random_agent_plots'))


env = make_env('9Bus')
env.seed(123456 + 0)
# selection of the best model 
model_path = f"{models_dir}/250000.zip"
model = PPO.load(model_path, env=env)

episodes = 24

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)
    fig, _ = env.plot_graph()
    fig.tight_layout(pad=0.1)
    fig.savefig(os.path.join(cwd, 'random_agent_plots', 'node_voltage_' + str(ep).zfill(4) + '.png'))
    plt.close()

images = []
filenames = sorted(glob.glob(os.path.join(cwd, "random_agent_plots/*.png")))
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave(os.path.join(cwd, 'random_agent_plots/node_voltage.gif'), images, fps=1)