import gym
import numpy as np
from gym.envs.toy_text import FrozenLakeEnv
from frozenlake_utils import plot_results
from utils import env_step, env_reset

seed = 42
env = FrozenLakeEnv(is_slippery=False)
env.seed(seed)
np.random.seed(seed)

num_actions = env.action_space.n
num_observations = env.observation_space.n

Q = np.zeros((num_observations, num_actions))

alpha = 3e-1
eps = 1.0
gamma = 0.9
alpha_decay = 0.999
eps_decay = 0.999
max_train_iterations = 10000
max_test_iterations = 100
max_episode_length = 200


def policy(state, is_training):
  # - with probability eps return a random action
  # - otherwise find the action that maximizes Q
  global eps
  if np.random.random() < eps and is_training:
    action = np.random.choice(num_actions)
  else:
    action = np.argmax(Q[state, :])
  return action


def train_step(state, action, reward, next_state, done):
  # - Q(s, a) <- Q(s, a) + alpha * (reward + gamma * Q(s', a') - Q(s, a))
  # - Make sure that Q(s', a') = 0 if we reach a terminal state
  global alpha
  if not done:
    Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
  else:
    Q[state, action] = Q[state, action] + alpha * (reward - Q[state, action])



def modify_reward(reward):
  # TODO: In some tasks, we will have to modify the reward.
  return reward


def run_episode(is_training):
  global eps
  episode_reward = 0
  state = env_reset(env, not is_training)
  #action = policy(state, is_training)
  for t in range(max_episode_length):
    action = policy(state, is_training)
    next_state, reward, done, _ = env_step(env, action, not is_training)
    reward = modify_reward(reward)
    episode_reward += reward
    #next_action = policy(next_state, is_training)
    if is_training:
      train_step(state, action, reward, next_state, done)
    state = next_state
    if done:
      break
  return episode_reward


# Training phase
train_reward = []
for it in range(max_train_iterations):
  train_reward.append(run_episode(is_training=True))
  alpha *= alpha_decay
  eps *= eps_decay

# Test phase
test_reward = []
for it in range(max_test_iterations):
  test_reward.append(run_episode(is_training=False))
plot_results(train_reward, test_reward, Q, env)
