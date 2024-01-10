import gym
from gym import spaces
import numpy as np
from config import *
import PD_environment as env


class GymInterface(gym.Env):
    def __init__(self):
        super(GymInterface, self).__init__()
        # Action space, observation space
        if RL_ALGORITHM == "PPO":
            # Define action space
            # Initialize the array for 6 actions (3 for center coordination, 3 for cutting plane angle)
            actionSize = 6
            # Create arrays for low and high values.
            # Use np.concatenate to combine the ranges for the two parts.
            low = np.concatenate(
                (np.full(3, ACTION_SPACE_CENTER_COOR_LOW), np.full(3, ACTION_SPACE_CUT_PLANE_ANGLE_LOW)))
            high = np.concatenate(
                (np.full(3, ACTION_SPACE_CENTER_COOR_HIGH), np.full(3, ACTION_SPACE_CUT_PLANE_ANGLE_HIGH)))
            # Define the action space as a Box object.
            action_space = spaces.Box(low=low, high=high, dtype=np.float32)

            # self.observation_space = spaces.Box(low=0, high=INVEN_LEVEL_MAX, shape=(len(I),), dtype=int)
            # Define observation space:
            os = [INVEN_LEVEL_MAX+1 for _ in range(len(I))]
            if STATE_DEMAND:
                os.append(DEMAND_QTY_MAX - DEMAND_QTY_MIN + 1)
            self.observation_space = spaces.MultiDiscrete(os)

        # Initialize the PD environment
        self.PD_env, self.PD_tree, self.decomposed_parts = env.create_env()
        self.total_reward_over_episode = []
        self.total_reward = 0
        self.num_episode = 1

    def reset(self):
        # Initialize the PD environment
        print("\nEpisode: ", self.num_episode)
        self.simpy_env, self.inventoryList, self.procurementList, self.productionList, self.sales, self.customer, self.providerList, self.daily_events = env.create_env(
            I, P, DAILY_EVENTS)
        env.simpy_event_processes(self.simpy_env, self.inventoryList, self.procurementList,
                                  self.productionList, self.sales, self.customer, self.providerList, self.daily_events, I)
        return env.cap_current_state(self.inventoryList)

    def step(self, action):
        # Update the action of the agent
        if RL_ALGORITHM == "DQN":
            I[1]["LOT_SIZE_ORDER"] = action
            # I[0]["DEMAND_QUANTITY"] = random.randint(DEMAND_QTY_MIN, DEMAND_QTY_MAX)
            # I[0]["DUE_DATE"] = random.randint(DUE_DATE_MIN, DUE_DATE_MAX)

        elif RL_ALGORITHM == "DDPG":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Raw Material":
                    # I[_]["LOT_SIZE_ORDER"] = int(round(action[i]))
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    i += 1
        elif RL_ALGORITHM == "PPO":
            i = 0
            for _ in range(len(I)):
                if I[_]["TYPE"] == "Raw Material":
                    I[_]["LOT_SIZE_ORDER"] = action[i]
                    i += 1

        # Capture the current state of the environment
        # current_state = env.cap_current_state(self.inventoryList)
        # Run the simulation for 24 hours (until the next day)
        self.simpy_env.run(until=self.simpy_env.now + 24)
        I[0]['DEMAND_QUANTITY'] = random.randint(
            DEMAND_QTY_MIN, DEMAND_QTY_MAX)
        # Capture the next state of the environment
        next_state = env.cap_current_state(self.inventoryList)
        # Calculate the total cost of the day
        daily_total_cost = env.cal_daily_cost_ACC(
            self.inventoryList, self.procurementList, self.productionList, self.sales)
        '''
        s = []
        for _ in range(len(self.inventoryList)):
            s.append(self.inventoryList[_].current_level)
        daily_total_cost = env.cal_daily_cost_DESC(s[0], s[1], action)
        '''
        if PRINT_SIM_EVENTS:
            # Print the simulation log every 24 hours (1 day)
            print(f"\nDay {(self.simpy_env.now+1) // 24}:")
            if RL_ALGORITHM == "DQN":
                print(f"[Order Quantity for {I[1]['NAME']}] ", action)
            else:
                i = 0
                for _ in range(len(I)):
                    if I[_]["TYPE"] == "Raw Material":
                        print(
                            f"[Order Quantity for {I[_]['NAME']}] ", action[i])
                        i += 1
            for log in self.daily_events:
                print(log)
            print("[Daily Total Cost] ", daily_total_cost)
            print("[STATE for the next round] ", next_state)
        self.daily_events.clear()
        reward = -daily_total_cost
        self.total_reward += reward
        # 현재 시뮬레이션(에피소드)이 종료되었는지에 대한 조건
        done = self.simpy_env.now >= SIM_TIME * 24  # 예: SIM_TIME일 이후에 종료
        if done == True:
            print("Total reward: ", self.total_reward)
            self.total_reward_over_episode.append(self.total_reward)
            self.total_reward = 0
            self.num_episode += 1

        info = {}  # 추가 정보 (필요에 따라 사용)

        # self.all_order_quantities.append(action)
        # self.all_rewards.append(reward)
        # self.all_inventory_levels.append((next_state[0], next_state[1]))
        return next_state, reward, done, info

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
