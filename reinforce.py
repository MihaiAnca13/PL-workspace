from collections import namedtuple, OrderedDict, deque
from copy import copy
import gym
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

GAMMA = 0.99
LEARNING_RATE = 1e-2
EPOCHS = 100
EPISODES_TO_TRAIN = 4
HIDDEN_SIZE = 128

Experience = namedtuple('Experience', ('state', 'action', 'reward', 'last_state'))


def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(copy(sum_r))
    return list(reversed(res))


class Model(nn.Module):
    def __init__(self, input_size, n_actions):
        super(Model, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class RLDataset(IterableDataset):
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    def __iter__(self):
        state = self.env.reset()

        states, actions, rewards, next_states = [], [], [], []
        while True:
            action = self.agent([state]).item()
            next_state, reward, done, _ = self.env.step(action)

            if done:
                states.append(np.array(state, dtype=np.float32))
                actions.append(action)
                rewards.append(reward)
                next_states.append(-1) # None not allowed...
                yield list(zip(states, actions, rewards, next_states))
                states, actions, rewards, next_states = [], [], [], []
                state = self.env.reset()
            else:
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                state = next_state


class Agent:
    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def __call__(self, states):
        np_states = np.array(states, dtype=np.float32)
        states_v = torch.tensor(np_states).to('cuda:0')
        probs_v = self.model(states_v)
        # converting logits to probabilities
        probs_v = F.softmax(probs_v, dim=1)
        probs = probs_v.data.cpu().numpy()
        # Converts probabilities of actions into action by sampling them
        actions = [np.random.choice(len(prob), p=prob, replace=False) for prob in probs]
        return np.array(actions)


class REINFORCE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.env = gym.make('CartPole-v0')
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = Model(obs_size, n_actions)
        self.agent = Agent(self.net)

        self.reward_100 = deque(maxlen=100)

    def forward(self, x):
        return self.net(x)

    def calculate_loss(self, batch):
        states, actions, qvals = batch
        logits_v = self.net(torch.cat(states).float())
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = torch.cat(qvals) * log_prob_v[list(range(len(states))), list(torch.cat(actions).cpu().numpy())]
        loss_v = -log_prob_actions_v.mean()
        return loss_v

    def training_step(self, batch, nb_batch):
        states, actions, rewards, next_states = zip(*batch)
        qvals = calc_qvals(rewards)
        loss = self.calculate_loss((states, actions, qvals))

        sum_r = sum(rewards)
        self.reward_100.append(sum_r)

        log = {
            'loss': loss,
            'reward': torch.tensor(sum_r).to('cuda:0'),
        }

        status = {
                  'reward_100': torch.tensor(sum(self.reward_100)/100.0).to('cuda:0')
                  }

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)
        return [optimizer]

    def __dataloader(self):
        dataset = RLDataset(self.env, self.agent)
        dataloader = DataLoader(dataset=dataset, batch_size=1)
        return dataloader

    def train_dataloader(self):
        return self.__dataloader()


if __name__ == '__main__':
    model = REINFORCE()

    trainer = pl.Trainer(gpus=1, max_epochs=EPOCHS)

    trainer.fit(model)
