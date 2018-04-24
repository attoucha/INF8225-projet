# based on https://github.com/Shmuma/ptan/tree/master/samples/rainbow

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



env = make_atari("PongNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, pytorch_img=True)


class ReplayMemory(object):
    def __init__(self, size):
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def push(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def sample(self, batch_size):
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)




memory = ReplayMemory(100000)





class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):

        weight_epsilon = self.weight_epsilon.cuda()
        bias_epsilon   = self.bias_epsilon.cuda()


        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(weight_epsilon).cuda())
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(bias_epsilon).cuda())
        else:
            weight = self.weight_mu
            bias   = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, num_actions, num_atoms, Vmin, Vmax):
        super(RainbowDQN, self).__init__()

        self.input_shape   = input_shape
        self.num_actions  = num_actions
        self.num_atoms    = num_atoms
        self.Vmin         = Vmin
        self.Vmax         = Vmax

        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.noisy_value1 = NoisyLinear(7*7*64, 512)
        self.noisy_value2 = NoisyLinear(512, self.num_atoms)

        self.noisy_advantage1 = NoisyLinear(7*7*64, 512)
        self.noisy_advantage2 = NoisyLinear(512, self.num_atoms * self.num_actions)

    def forward(self, x):
        batch_size = x.size(0)

        x = x / 255.
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(batch_size, -1)

        value = F.relu(self.noisy_value1(x))
        value = self.noisy_value2(value)

        advantage = F.relu(self.noisy_advantage1(x))
        advantage = self.noisy_advantage2(advantage)

        value     = value.view(batch_size, 1, self.num_atoms)
        advantage = advantage.view(batch_size, self.num_actions, self.num_atoms)

        x = value + advantage - advantage.mean(1, keepdim=True)
        x = F.softmax(x.view(-1, self.num_atoms)).view(-1, self.num_actions, self.num_atoms)
        return x

    def reset_noise(self):
        self.noisy_value1.reset_noise()
        self.noisy_value2.reset_noise()
        self.noisy_advantage1.reset_noise()
        self.noisy_advantage2.reset_noise()


num_atoms = 51
Vmin = -10
Vmax = 10



class Agent(object):

    def __init__(self):
        self.Q = RainbowDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
        self.target_Q = RainbowDQN(env.observation_space.shape, env.action_space.n, num_atoms, Vmin, Vmax)
        self.optimizer = optim.Adam(self.Q.parameters(), lr=0.0000625)
        self.num_atoms = num_atoms
        self.Vmin = Vmin
        self.Vmax = Vmax
        self.Q.cuda()
        self.target_Q.cuda()
        hard_update(self.target_Q, self.Q)


    def act(self, state):

        state = Variable(torch.FloatTensor(np.float32(state)).unsqueeze(0), volatile=True).cuda()

        
        dist = self.Q(state).data.cpu()
        dist = dist * torch.linspace(self.Vmin, self.Vmax, self.num_atoms)
        action = dist.sum(2).max(1)[1].numpy()[0]

        return action

    def backward(self, batch_size):

        state, action, reward, next_state, done = memory.sample(batch_size)
        state      = Variable(torch.FloatTensor(np.float32(state))).cuda()
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), volatile=True).cuda()
        action     = Variable(torch.LongTensor(action)).cuda()
        reward     = torch.FloatTensor(reward)
        done       = torch.FloatTensor(np.float32(done))

        proj_dist = self.projection_distribution(next_state, reward, done)

        dist = self.Q(state)
        action = action.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, num_atoms)
        dist = dist.gather(1, action).squeeze(1)
        dist.data.clamp_(0.01, 0.99)
        loss = -(Variable(proj_dist).cuda() * dist.log()).sum(1)
        loss  = loss.mean()

        # backpropagation of loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.Q.reset_noise()
        self.target_Q.reset_noise()

        return loss


    def projection_distribution(self,next_state, rewards, dones):
        batch_size  = next_state.size(0)

        delta_z = float(Vmax - Vmin) / (num_atoms - 1)
        support = torch.linspace(Vmin, Vmax, num_atoms)

        next_dist   = self.target_Q(next_state).data.cpu() * support
        next_action = next_dist.sum(2).max(1)[1]
        next_action = next_action.unsqueeze(1).unsqueeze(1).expand(next_dist.size(0), 1, next_dist.size(2))
        next_dist   = next_dist.gather(1, next_action).squeeze(1)

        rewards = rewards.unsqueeze(1).expand_as(next_dist)
        dones   = dones.unsqueeze(1).expand_as(next_dist)
        support = support.unsqueeze(0).expand_as(next_dist)

        Tz = rewards + (1 - dones) * 0.99 * support
        Tz = Tz.clamp(min=Vmin, max=Vmax)
        b  = (Tz - Vmin) / delta_z
        l  = b.floor().long()
        u  = b.ceil().long()

        offset = torch.linspace(0, (batch_size - 1) * num_atoms, batch_size).long()\
                    .unsqueeze(1).expand(batch_size, num_atoms)

        proj_dist = torch.zeros(next_dist.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))

        return proj_dist


def soft_update(target, source, tau):

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data*(1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)



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

    while True:

        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        memory.push(state, action, np.sign(reward), next_state, done)

        state = next_state
        total_reward += reward
        steps += 1
        total_step += 1
        if total_step > 1950000:
            frame = env.render(mode='rgb_array')
            frames.append(frame)

        if len(memory) >= 10000:
            agent.backward(batch_size)

        if total_step % 10000 == 0:
            hard_update(agent.target_Q, agent.Q)


        if done:
            print(" Episode: {0} finished after {1}: steps  score : {2} total steps : {3} total time : {4} ".format(episode, steps,total_reward,total_step,int(time.time() - start_time)))
            break

    if total_step >= 2*10**6:
        break

    rewards.append(total_reward)






pickle_out = open("rain.pickle","wb")
pickle.dump(rewards, pickle_out)
pickle_out.close()



print("--- %s seconds ---" % (time.time() - start_time))

display_frames_as_gif(frames, filename = "rain.mp4")

