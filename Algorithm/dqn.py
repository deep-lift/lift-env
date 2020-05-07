from tensorboardX import SummaryWriter
import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# 라이브러리 불러오기
from mlagents.envs import UnityEnvironment
# import tensorflow as tf
import random
import datetime
from collections import deque
import numpy as np
import torch

n_agent = 4
agent_observation_size = 46
state_size = 46*4
action_size = 3
hidden_size = 200

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 500000
discount_factor = 0.99
tau = 1e-3

start_train_episode = 10
run_episode = 3000000
test_episode = 10000

print_interval = 10
save_interval = 1000

date_time = datetime.datetime.now().strftime('%Y%m%d-%H-%M-%S')

game = '../Env'
env_name = game + "/Elevator"

# save_path = '../saved_models/' + game + '/' + date_time + '_DDPG'
# load_path = '../saved_models/' + game + '/' + 'INSERT_DATETIME_TO_LOAD' + '_DDPG'

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
num_layers = 2

summary = SummaryWriter()


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.dense1 = nn.Linear(agent_observation_size, hidden_size)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, action_size)

    def forward(self, x):  # 4 * 46
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5000 * 10000
TARGET_UPDATE = 10

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

policy_net = DQN().to(device).double()
target_net = DQN().to(device).double()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters(),lr=0.003)
memory = ReplayMemory(mem_maxlen)

steps_done = 0


def select_action(state, needed_action_agents):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # print('network')
            # return policy_net(state).squeeze(0).max(1)[1]
            return policy_net(state).max(1).indices.tolist(), eps_threshold
    else:
         return [random.randrange(action_size) for i in needed_action_agents], eps_threshold



episode_durations = []


def plot_durations():
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.double)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated


def optimize_model(episode):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool).to(device)
    non_final_next_states = torch.stack([s for s in batch.next_state if s is not None]).to(device).double()
    state_batch = torch.stack(batch.state).to(device).double()
    action_batch = torch.tensor(np.stack(batch.action)).to(device)
    reward_batch = torch.tensor(np.stack(batch.reward)).to(device).double()

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(-1).long())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32).double()
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values.detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values.squeeze().reshape(-1), expected_state_action_values)
    summary.add_scalar('loss/critic_loss', loss.item(), episode)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


# Main 함수 -> DDPG 에이전트를 드론 환경에서 학습
if __name__ == '__main__':
    env = UnityEnvironment(file_name=env_name)
    default_brain = env.brain_names[0]
    rewards = deque(maxlen=print_interval)
    success_cnt = 0
    step = 0

    # 각 에피소드를 거치며 replay memory에 저장
    for episode in range(run_episode + test_episode):
        if episode == run_episode:
            train_mode = False

        env_info = env.reset(train_mode=train_mode)[default_brain]
        state = env_info.vector_observations
        l_agents = env_info.agents
        episode_rewards = 0
        done = False
        step = 0

        while not done:
            step += 1
            needed_action_agents = [l_agents.index(i) for i in env_info.agents]
            states = np.split(state[0], 4)
            filtered_state = []
            for i in needed_action_agents:
                filtered_state.append(states[i])
            states = torch.tensor(filtered_state).to(device).double()
            actions, eps_threshold = select_action(states, needed_action_agents)
            env_info = env.step(actions)[default_brain]
            done = env.global_done
            r = env_info.rewards[0]
            episode_rewards += r

            if not done:
                ns = env_info.vector_observations
                next_states = np.split(ns[0], 4)
                filtered_next_state = []
                for i in needed_action_agents:
                    filtered_next_state.append(next_states[i])
                ns = torch.tensor(filtered_next_state).to(device).double()
            else:
                ns = [None for _ in needed_action_agents]
                summary.add_scalar('reward/episode_rewards', episode_rewards, episode)
                summary.add_scalar('parameter/eps_threshold', eps_threshold, episode)

            # print('step : {}, state : {}, reward : {},  next_state: {}, done : {}'.format(step, s.shape, r, ns.shape, done))

            if train_mode:
                for i, _ in enumerate(needed_action_agents):
                    memory.push(states[i], actions[i], ns[i], r)

            s = ns

            # train_mode 이고 일정 이상 에피소드가 지나면 학습
            if episode > start_train_episode and train_mode and done:
                optimize_model(episode)

        success_cnt = success_cnt + 1 if r >= 0. else success_cnt
        rewards.append(episode_rewards)

        # 일정 이상의 episode를 진행 시 log 출력
        if episode % print_interval == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.3f} / success_cnt: {}".format(step, episode, np.mean(rewards), success_cnt))
            success_cnt = 0
            summary.add_scalar('reward/reward_mean', np.mean(rewards), episode)

        # 일정 이상 episode를 진행 시 현재 모델 저장
        if train_mode and episode % save_interval == 0 and episode != 0:
            print("model saved")
            # agent.save_model()

    env.close()
