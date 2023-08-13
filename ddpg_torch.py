import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=0.2, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)


class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.mem_cntr += 1

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - done

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminal

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name

        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)
        f3 = 0.003
        T.nn.init.uniform_(self.q.weight.data, -f3, f3)
        T.nn.init.uniform_(self.q.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = F.relu(self.action_value(action))

        state_action_value = T.add(state_value, action_value)
        state_action_value = F.relu(state_action_value)
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_file = f"{checkpoint_dir}/{self.name}.torch"
        T.save(self.state_dict(), checkpoint_file)
        print(f"saving {checkpoint_file}")

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_file = f"{checkpoint_dir}/{self.name}.torch"
        self.load_state_dict(T.load(checkpoint_file))
        print(f"loading {checkpoint_file}")

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(ActorNetwork, self).__init__()

        self.alpha = alpha
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name

        self.fc1 = nn.Linear(*input_dims, self.fc1_dims)
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)

        self.bn1 = nn.LayerNorm(self.fc1_dims)

        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)

        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, n_actions)
        f3 = 0.003
        T.nn.init.uniform_(self.mu.weight.data, -f3, f3)
        T.nn.init.uniform_(self.mu.bias.data, -f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=self.alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        state_value = F.relu(state_value)
        state_value = self.mu(state_value)
        state_value = T.tanh(state_value)

        return state_value

    def save_checkpoint(self, checkpoint_dir):
        checkpoint_file = f"{checkpoint_dir}/{self.name}.torch"
        T.save(self.state_dict(), checkpoint_file)
        print(f"saving {checkpoint_file}")

    def load_checkpoint(self, checkpoint_dir):
        checkpoint_file = f"{checkpoint_dir}/{self.name}.torch"
        self.load_state_dict(T.load(checkpoint_file))
        print(f"loading {checkpoint_file}")


class Agent(object):
    def __init__(self, alpha, beta, input_dims, tau, gamma=0.99, n_actions=2,
                 max_size=1000000, layer1_size=400, layer2_size=300, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions, 'actor')
        self.target_actor = ActorNetwork(alpha, input_dims, layer1_size, layer2_size,
                                  n_actions, 'target_actor')

        self.critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                  n_actions, 'critic')
        self.target_critic = CriticNetwork(beta, input_dims, layer1_size, layer2_size,
                                  n_actions, 'target_critic')

        self.noise = OUActionNoise(np.zeros(n_actions))

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor.eval()
        observation = T.tensor(observation, dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(observation).to(self.actor.device)
        mu_prime = mu + T.tensor(self.noise(), dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            print(f"{self.memory.mem_cntr} is not enough samples, not learning until we have {self.batch_size}...")
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        new_state = T.tensor(new_state, dtype=T.float).to(self.critic.device)
        done = T.tensor(done, dtype=T.float).to(self.critic.device)

        self.target_actor.eval()
        self.critic.eval()
        self.target_critic.eval()

        target_actions = self.target_actor.forward(new_state)
        target_critic_value = self.target_critic.forward(new_state, target_actions)
        critic_value = self.critic.forward(state, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * target_critic_value[j] * done[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor.forward(state)
        self.actor.train()
        actor_loss = -self.critic.forward(state, mu)
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_parameters = self.actor.named_parameters()
        critic_parameters = self.critic.named_parameters()
        target_actor_parameters = self.target_actor.named_parameters()
        target_critic_parameters = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_parameters)
        critic_state_dict = dict(critic_parameters)
        target_actor_state_dict = dict(target_actor_parameters)
        target_critic_state_dict = dict(target_critic_parameters)

        for name in actor_state_dict:
            actor_state_dict[name] = tau     * actor_state_dict[name].clone() + \
                                     (1-tau) * target_actor_state_dict[name].clone()
        self.target_actor.load_state_dict(actor_state_dict)

        for name in critic_state_dict:
            critic_state_dict[name] = tau    * critic_state_dict[name].clone() + \
                                     (1-tau) * target_critic_state_dict[name].clone()
        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.actor.save_checkpoint(checkpoint_dir)
        self.target_actor.save_checkpoint(checkpoint_dir)
        self.critic.save_checkpoint(checkpoint_dir)
        self.target_critic.save_checkpoint(checkpoint_dir)

    def load_models(self, checkpoint_dir):
        self.actor.load_checkpoint(checkpoint_dir)
        self.target_actor.load_checkpoint(checkpoint_dir)
        self.critic.load_checkpoint(checkpoint_dir)
        self.target_critic.load_checkpoint(checkpoint_dir)

