import torch
import torch.nn as nn
import json
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_params(params_dir):
    with open(params_dir) as fp:
        params = json.load(fp)
    return params


class A2C_model(nn.Module):
    def __init__(self, params_dir, input_dim , act_size):
        super().__init__()
        self.params = parse_params(params_dir)
        self.fc1 = nn.Linear(input_dim , self.params['hidden_dim'])
        self.actor_fc = nn.Linear(self.params['hidden_dim'],
                                  self.params['hidden_dim'])
        self.actor_out = nn.Linear(self.params['hidden_dim'], act_size)
        self.std = nn.Parameter(torch.ones(1, act_size))
        self.critic_fc = nn.Linear(self.params['hidden_dim'],
                                   self.params['hidden_dim'])
        self.critic_out = nn.Linear(self.params['hidden_dim'], 1)
        
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        mean = self.actor_out(F.relu(self.actor_fc(x)))
        dist = torch.distributions.Normal(mean, self.std)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        value = self.critic_out(F.relu(self.critic_fc(x)))
        return torch.clamp(action, -1, 1), log_prob, value


class Agent_A2c():
    def __init__(self, device, num_agents, params_dir, state_size, action_size):
        self.model = A2C_model(params_dir, state_size, action_size).to(device)
        self.device = device
        self.num_agents = num_agents
        self.params = self.model.params
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=self.params['lr'])

    def __call__(self, states):
        # mu, std, val, etp = self.model(states)
        actions, log_prob, val = self.model(states)
        return actions, log_prob, val

    def step(self, experiences):
        '''
            experiences:
                    actions (num agents * num actions)
                    rewards (size = num agents)
                    log_probs (num agents * num actions)
                    not_dones (size = num agents)
                    state_values (size = num agents)
        '''
        
        actions, rewards, log_probs, not_dones, state_values = experiences
        rewards = torch.FloatTensor(rewards).transpose(0, 1).contiguous()                      
        processed_experience = [None] * (len(experiences[0]) - 1)
        # MDP property
        return_  = state_values[-1].detach()
        for i in reversed(range(len(experiences[0])-1)):
            not_done_ = torch.FloatTensor(not_dones[i+1]).to(device).unsqueeze(1)
            reward_ = torch.FloatTensor(rewards[:,i]).to(device).unsqueeze(1)
            return_ = reward_ + self.params['gamma'] * not_done_ * return_
            next_value_ = state_values[i+1]
            advantage_  = reward_ + self.params['gamma'] * not_done_ * next_value_.detach() - state_values[i].detach()
            processed_experience[i] = [log_probs[i], advantage_, state_values[i], return_]
        log_probs, advantages, values, returns = map(
                lambda x: torch.cat(x, dim=0), zip(*processed_experience))
        policy_loss = -log_probs * advantages
        value_loss = 0.5 * (returns - values).pow(2)
        self.optimizer.zero_grad()
        loss = (policy_loss + value_loss).mean()
        # In case the model is not stable
        if torch.isnan(loss).any():
            print('Nan in loss function')
            pass
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(),
                                 self.model.params['grad_clip'])
        self.optimizer.step()

    
    
class Experience():
    def __init__(self):
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.not_dones = []
        self.state_values = []

    def add(self, actions, rewards, log_probs, not_dones, state_values):
        self.actions.append(actions)
        self.rewards.append(rewards)
        self.log_probs.append(log_probs)
        self.not_dones.append(not_dones)
        self.state_values.append(state_values)

    def spit(self):
        return (self.actions, self.rewards, self.log_probs, self.not_dones,
                self.state_values)