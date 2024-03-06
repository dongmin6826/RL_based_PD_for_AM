import gym
from gym import spaces
import numpy as np
from config import *
import PD_environment as env
from torch.utils.tensorboard import SummaryWriter


class GymInterface(gym.Env):
    def __init__(self):
        if TRAIN:
            self.writer = SummaryWriter(log_dir="logs/")
        super(GymInterface, self).__init__()
        # Initialize the PD environment
        self.PD_tree, self.decomposed_parts = env.create_env()
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.num_episode = 1

        # Define observation space:
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(MAX_N_PARTS*2, 6))
        self.update_state()

        self.Advantage = self.PD_tree[1]["SupVol"]
        # Define Vector Limits
        self.x = self.PD_tree[1]["Mesh"].vertices[:, 0]
        self.y = self.PD_tree[1]["Mesh"].vertices[:, 1]
        self.z = self.PD_tree[1]["Mesh"].vertices[:, 2]

        # # Define action space
        self.define_action_space()

    # Update action_space at every step
    def define_action_space(self):

        # Update Action Space Low Values
        self.action_space_low = np.concatenate([
            [0],
            [np.min(self.x, axis=0)],
            [np.min(self.y, axis=0)],
            [np.min(self.z, axis=0)],
            [np.min(self.x, axis=0)],
            [np.min(self.y, axis=0)],
            [np.min(self.z, axis=0)],
        ])

        # Update Action Space High Values
        self.action_space_high = np.concatenate([
            [len(self.decomposed_parts)-1],
            [np.max(self.x, axis=0)],
            [np.max(self.y, axis=0)],
            [np.max(self.z, axis=0)],
            [np.max(self.x, axis=0)],
            [np.max(self.y, axis=0)],
            [np.max(self.z, axis=0)],
        ])

        # Update Action Space
        self.action_space = spaces.Box(
            low=self.action_space_low, high=self.action_space_high, shape=(7,), dtype=np.float32)

    # Update State
    def update_state(self):
        print("Parts_List:", self.decomposed_parts)
        # Make Observation Space
        cont = 0  # First Dimension Coordinate
        self.current_observation = np.zeros((MAX_N_PARTS*2, 6))
        for part_id in self.decomposed_parts:
            cont2 = 0  # Second Dimension Coordinate

            for key in self.PD_tree[1].keys():
                if key != "Mesh":
                    self.current_observation[cont][cont2] = (
                        self.PD_tree[part_id][key])
                cont2 = cont2+1
            cont = cont+1

    def reset(self):
        # Initialize the PD environment
        print("\nEpisode: ", self.num_episode)
        self.PD_tree, self.decomposed_parts = env.create_env()

        self.define_action_space()
        self.update_state()

        # return env.cap_current_state(self.PD_tree, self.decomposed_parts)
        return self.current_observation

    # def step(self, action):
    #     # Update the action of the agent
    #     self.PD_tree, self.decomposed_parts = env.decompose_parts(
    #         action)
    #     # Capture the next state of the environment
    #     next_state = env.cap_current_state(
    #         self.PD_tree, self.decomposed_parts)
    #     # Calculate the reward
    #     reward = env.cal_reward(next_state)
    #     self.total_reward += reward
    #     # 한 에피소드 종료 조건
    #     if MAX_N_PARTS == len(self.decomposed_parts):
    #         done = True
    #     if done == True:
    #         print("Total reward: ", self.total_reward)
    #         self.total_reward_over_episode.append(self.total_reward)
    #         self.total_reward = 0
    #         self.num_episode += 1

    #     info = {}  # 추가 정보 (필요에 따라 사용)
    #     return next_state, reward, done, info
    def step(self, action):
        done = False  # Stop Learing Variable

        # Take an action
        PD_tree, decomposed_parts, reward = env.decompose_parts(
            action, self.decomposed_parts, self.PD_tree)

        if len(decomposed_parts) > MAX_N_PARTS*2:
            obs, reward, done, _ = env.step(action)
            return obs, reward, done, _
        else:
            self.PD_tree, self.decomposed_parts = PD_tree, decomposed_parts
            # Update Action Space
            self.define_action_space()

            # Capture the next state of the environment
            self.update_state()

            # Conditions for ending one episode
            if MAX_N_PARTS < len(self.decomposed_parts) or reward == 0:
                done = True

            # Calculate the reward
            reward = -reward
            self.total_reward = -reward
            if done == True:
                if TRAIN:
                    self.writer.add_scalar(
                        "reward", reward, global_step=self.num_episode)
                print("Total reward: ", self.total_reward)
                self.total_reward_over_episode.append(self.total_reward)
                self.total_reward = 0
                self.num_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)

        return self.current_observation, reward, done, info

    def render(self, mode='human'):
        pass
        # if EPISODES == 1:
        #     self.visualize()
        # else:
        #     if OPTIMIZE_HYPERPARAMETERS:
        #         pass
        #     else:
        #         self.visualize()

    def close(self):
        # 필요한 경우, 여기서 리소스를 정리
        pass
