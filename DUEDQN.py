import numpy as np
import time
import gym
import pickle
import math, random
import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
from torch.autograd import Variable
from  matplotlib import animation
from wrappers import make_atari, wrap_deepmind
from ipywidgets import widgets
from JSAnimation.IPython_display import display_animation




class ReplayMemory(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)




env = make_atari("PongNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, pytorch_img=True)

memory = ReplayMemory(100000)


class DuelingDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DuelingDQN, self).__init__()

        self.input_shape = input_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(),
            nn.Linear(512, 1))

    def forward(self, x):

        x = self.features(x)
        x = x.view(x.size(0), -1)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()


class Agent(object):

    def __init__(self):
        self.Q = DuelingDQN(env.observation_space.shape, env.action_space.n)
        self.target_Q = DuelingDQN(env.observation_space.shape, env.action_space.n)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.0000625)
        hard_update(self.target_Q, self.Q)
        self.Q.cuda()
        self.target_Q.cuda()

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0), volatile=True).cuda()
            q_value = self.Q(state)
            return q_value.max(1)[1].data[0]
        else:
            return random.randrange(env.action_space.n)

    def backward(self, batch_size):

        state, action, reward, next_state, done = memory.sample(batch_size)
        state = Variable(torch.FloatTensor(np.float32(state))).cuda()
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True).cuda()
        action = Variable(torch.LongTensor(action)).cuda()
        reward = Variable(torch.FloatTensor(reward)).cuda()
        done = Variable(torch.FloatTensor(done)).cuda()

        # current Q values
        q_values      = self.Q(state)
        q_value       = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # next Q values
        next_q_values = self.Q(next_state)
        next_q_state_values = self.target_Q(next_state)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)



        #  Compute the expected Q

        expected_q_value = reward + gamma * next_q_value * (1 - done)

        loss = (q_value - Variable(expected_q_value.data)).pow(2).mean()
    # backpropagation of loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss


def display_frames_as_gif(frames, filename = None):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(filename, writer = None, fps=50)

def soft_update(target, source, tau):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


agent = Agent()

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 30000

epsilon_by_step = lambda step: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step / epsilon_decay)


batch_size = 32
gamma = 0.99

rewards = []
frames = []
total_step = 0
episode = 0

start_time = time.time()

while True:
    episode += 1
    steps = 0
    state = env.reset()
    done = False
    total_reward = 0
    epsilon = epsilon_by_step(total_step)

    while True:
        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, np.sign(reward), next_state, done)

        state = next_state
        total_reward += reward
        steps += 1
        total_step += 1
        
        if total_step > 1950000:
            frame = env.render(mode='rgb_array')
            frames.append(frame)

        if len(memory) >= 50000:
            agent.backward(batch_size)
        
        if total_step % 10000 == 0:
            hard_update(agent.target_Q, agent.Q)


        if done:
            print(" Episode: {0} finished after {1}: steps  score : {2} total steps : {3} total time : {4} ".format(episode, steps,total_reward,total_step,int(time.time() - start_time)))
            break

    rewards.append(total_reward)
    

    if total_step >= 2*10**6:
        break


pickle_out = open("dueling.pickle","wb")
pickle.dump(rewards, pickle_out)
pickle_out.close()



print("--- %s seconds ---" % (time.time() - start_time))

display_frames_as_gif(frames, filename = "ddqn.mp4")
