#!/usr/bin/env python
# coding: utf-8

# # Coursework Template

# ## Setup

# **Dependencies and imports**
# 
# This can take a minute...

# In[1]:


# !pip install swig
# !pip install rldurham


# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # disable CUDA (better on Colab/NCC: choose an environment without GPU)
import torch
import rldurham as rld
import matplotlib
matplotlib.use('Agg')  # 不弹出窗口
import matplotlib.pyplot as plt

os.makedirs("graphs", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("videos", exist_ok=True)

# **Reinforcement learning agent**
# 
# Replace this with your own agent. Also see the SAC implementationi below for a starting point.

# In[3]:


class Agent(torch.nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.discrete_act, self.discrete_obs, self.act_dim, self.obs_dim = rld.env_info(env)

    def sample_action(self, s):
        return torch.rand(self.act_dim) * 2 - 1 # unifrom random in [-1, 1]

    def put_data(self, action, observation, reward):
        pass

    def train(self):
        pass


# ## Training

# **Prepare the environment and wrap it to capture statistics, logs, and videos**

# In[4]:


env = rld.make("rldurham/Walker", render_mode="rgb_array")
# env = rld.make("rldurham/Walker", render_mode="rgb_array", hardcore=True) # only attempt this when your agent has solved the non-hardcore version

# get statistics, logs, and videos
env = rld.Recorder(
    env,
    smoothing=10,                       # track rolling averages (useful for plotting)
    video=True,                         # enable recording videos
    video_folder="videos",              # folder for videos
    video_prefix="cqst66-agent-video",  # prefix for videos (replace xxxx00 with your username)
    logs=True,                          #    keep logs
)

# training on CPU recommended
rld.check_device()

# environment info
rld.env_info(env, print_out=True)

# render start image (reset just to render image)
env.reset(seed=42)
rld.render(env)


# **Collect episodes and train the agent**

# In[5]:


# in the submission please use seed_everything with seed 42 for verification
seed, observation, info = rld.seed_everything(42, env)

# initialise agent
agent = Agent(env)
max_episodes = 10

# track statistics for plotting
tracker = rld.InfoTracker()

# switch video recording off (only switch on every x episodes as this is slow)
env.video = False

# training procedure
for episode in range(max_episodes):
    
    # recording statistics and video can be switched on and off (video recording is slow!)
    env.info = True                 # usually tracking every episode is fine
    env.video = episode % 100 == 0  # record videos every 100 episodes (set BEFORE calling reset!)

    # reset for new episode
    observation, info = env.reset()

    # run episode
    done = False
    while not done:
        
        # select the agent action
        action = agent.sample_action(observation)

        # take action in the environment
        observation, reward, terminated, truncated, info = env.step(action)

        # remember
        agent.put_data(action, observation, reward)

        # check whether done
        done = terminated or truncated

    # train the agent after each episode
    agent.train()
            
    # track and plot statistics
    tracker.track(info)
    if (episode + 1) % 10 == 0:
        tracker.plot(r_mean_=True, r_std_=True, r_sum=dict(linestyle=':', marker='x'))

# don't forget to close environment (e.g. triggers last video save)
env.close()

# write log file (for coursework)
env.write_log(folder="logs", file="cqst66-agent-log.txt")  # replace xxxx00 with your username


# ## Heuristic demo

# A small demo with a predefined heuristic that is suboptimal and has no notion of balance (designed for the orignal BipedalWalker environment, but should also work fine for rldurham/Walker).

# In[6]:


from gymnasium.envs.box2d.bipedal_walker import BipedalWalkerHeuristics

env = rld.make(
    "rldurham/Walker",
    # "BipedalWalker-v3",
    render_mode="human",
    # render_mode="rgb_array",
    hardcore=False,
    # hardcore=True,
)
_, obs, info = rld.seed_everything(42, env)

heuristics = BipedalWalkerHeuristics()

act = heuristics.step_heuristic(obs)
for _ in range(500):
    obs, rew, terminated, truncated, info = env.step(act)
    act = heuristics.step_heuristic(obs)
    if terminated or truncated:
        break
    if env.render_mode == "rgb_array":
        rld.render(env, clear=True)
env.close()


# ## SAC Implementation

# This is a Soft-Actor-Critic (SAC) implementation (adapted from https://github.com/seungeunrho/minimalRL/blob/master/sac.py) that does not perform very well, but is a decent starting point for the coursework. It can be outperformed by simpler methods (e.g. TD3) if they are well tuned and you are free to use any other method. Overall, the task is to make use of various ideas and principles that were covered in the lecture and/or that you find in papers to achieve the best possible performance.

# In[7]:


import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import collections, random


# In[8]:

class ReplayBuffer():
    def __init__(self, buffer_limit):
        self.buffer = collections.deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, terminated  = transition
            s_lst.append(s)
            a_lst.append(a)
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if terminated else 1.0 
            done_mask_lst.append([done_mask])

        return torch.tensor(np.array(s_lst), dtype=torch.float), \
               torch.tensor(np.array(a_lst), dtype=torch.float), \
               torch.tensor(np.array(r_lst), dtype=torch.float), \
               torch.tensor(np.array(s_prime_lst), dtype=torch.float), \
               torch.tensor(np.array(done_mask_lst), dtype=torch.float)
    
    def size(self):
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, act_dim)

        # 最后一层用小权重初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, x):
        x = F.relu(self.ln1(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Critic, self).__init__()
        # Q1
        self.fc1 = nn.Linear(obs_dim + act_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)

        # Q2
        self.fc4 = nn.Linear(obs_dim + act_dim, 400)
        self.ln2 = nn.LayerNorm(400)
        self.fc5 = nn.Linear(400, 300)
        self.fc6 = nn.Linear(300, 1)

        # 最后一层小权重初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.fc6.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc6.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        q1 = F.relu(self.ln1(self.fc1(x)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)

        q2 = F.relu(self.ln2(self.fc4(x)))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)

        return q1, q2

    def q1_forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.ln1(self.fc1(x)))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

class TD3Agent(nn.Module):
    def __init__(self,
                 env,
                 lr_actor=3e-4,
                 lr_critic=3e-4,
                 buffer_limit=1000000,
                 gamma=0.99,
                 batch_size=256,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 exploration_noise=0.1):
        super().__init__()
        
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.total_it = 0
        
        discrete_act, discrete_obs, act_dim, obs_dim = rld.env_info(env, print_out=True)
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        
        self.actor = Actor(obs_dim, act_dim)
        self.actor_target = Actor(obs_dim, act_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        
        self.critic = Critic(obs_dim, act_dim)
        self.critic_target = Critic(obs_dim, act_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.memory = ReplayBuffer(buffer_limit=buffer_limit)
    
    def action(self, state, add_noise=True):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)

        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]

        if add_noise:
            noise = np.random.normal(0, self.exploration_noise, size=self.act_dim)
            action = action + noise
        
        action = np.clip(action, -1.0, 1.0)

        return action
    
    def put(self, s, a, r, s_prime, done):
        self.memory.put((s, a, r, s_prime, done)) 
    def update(self):
        if self.memory.size() < self.batch_size:
            return
        
        self.total_it += 1
        state, action, reward, next_state, done_mask = self.memory.sample(self.batch_size)
        
        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + self.gamma * done_mask * target_q
        
        current_q1, current_q2 = self.critic(state, action)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1_forward(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            self.soft_update(self.actor, self.actor_target)
            self.soft_update(self.critic, self.critic_target)
    
    def soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def episode(self, n_epi=0):
        done = False
        s, _ = self.env.reset()
        info = {}
        total_reward = 0
        while not done:
            a = self.action(s, add_noise=True)
            if n_epi > 700:
                a = a * 0.95
            s_prime, r, terminated, truncated, info = self.env.step(a)
            total_reward += r
            done = terminated or truncated
            self.put(s, a, r, s_prime, terminated)
            self.update()
            s = s_prime
        info['total_reward'] = total_reward
        return info

# In[12]:


env = rld.make("rldurham/Walker", render_mode="rgb_array")
rld.seed_everything(42, env)

env = rld.Recorder(env, smoothing=10, video=True, logs=True,
                   video_folder="videos", video_prefix="cqst66-agent-video")
recorder = env
recorder.video = False
tracker = rld.InfoTracker()
env = rld.transparent_wrapper(gym.wrappers.ClipReward)(env, min_reward=-10)

agent = TD3Agent(env)

for n_epi in range(1000):
    if n_epi > 700:
        agent.exploration_noise = 0.05 
    recorder.video = (n_epi + 1) % 100 == 0
    info = agent.episode(n_epi)
    env.add_stats(info, ignore_existing=True) 
    tracker.track(info)
    print(f"Episode: {n_epi + 1}/1000, Score: {info['recorder']['r_sum']:.2f}", flush=True)

    if (n_epi + 1) % 100 == 0:  
        tracker.plot(r_mean_=True, r_std_=True, r_sum=dict(linestyle=':', marker='x'))
        plt.savefig(f"graphs/progress_{n_epi + 1}.png")
        plt.close()

env.close() 
env.write_log(folder="logs", file="cqst66-agent-log.txt")
print("saved")