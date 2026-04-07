import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_race.envs.pyrace_2d import PyRace2D

class RaceEnv(gym.Env):
    metadata = {'render_modes' : ['human'], 'render_fps' : 30}
    def __init__(self, render_mode="human", ):
        print("init")
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(np.array([0, 0, 0, 0, 0]), np.array([10, 10, 10, 10, 10]), dtype=int)
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.memory = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs=[]
        self.pyrace = PyRace2D(self.is_view, mode = self.render_mode)
        obs = self.pyrace.observe()
        return np.array(obs),{}

    def step(self, action):
        self.pyrace.action(action)
        reward = self.pyrace.evaluate()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe()
        return np.array(obs), reward, done, False, {'dist':self.pyrace.car.distance, 'check':self.pyrace.car.current_check, 'crash': not self.pyrace.car.is_alive}

    # def render(self, close=False , msgs=[], **kwargs): # gymnasium.render() does not accept other keyword arguments
    def render(self): # gymnasium.render() does not accept other keyword arguments
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def save_memory(self, file):
        # print(self.memory) # heterogeneus types
        # np.save(file, self.memory)
        np.save(file, np.array(self.memory, dtype=object))
        print(file + " saved")

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


class RaceEnvV3(gym.Env):
    """
    Pyrace-v3 improvements over v1:
    Observations: 6 continuous floats in [0,1]
    Actions: 4 (ACCELERATE, TURN_LEFT, TURN_RIGHT, BRAKE)
    Reward: Dense reward function (progress + speed bonus + checkpoint/goal milestones)
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode="human"):
        print("init RaceEnvV3")
        # 4 actions
        self.action_space = spaces.Discrete(4)
        # 5 radar readings + 1 speed, all normalised to [0, 1]
        self.observation_space = spaces.Box(low=np.zeros(6, dtype=np.float32), high=np.ones(6, dtype=np.float32), dtype=np.float32,)
        self.is_view = True
        self.pyrace = PyRace2D(self.is_view)
        self.msgs = []
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        mode = self.pyrace.mode
        del self.pyrace
        self.is_view = True
        self.msgs = []
        self.pyrace = PyRace2D(self.is_view, mode=self.render_mode)
        obs = self.pyrace.observe_v3()
        return np.array(obs, dtype=np.float32), {}

    def step(self, action):
        self.pyrace.action_v3(action)
        reward = self.pyrace.evaluate_v3()
        done   = self.pyrace.is_done()
        obs    = self.pyrace.observe_v3()
        info   = {
            'dist':  self.pyrace.car.distance,
            'check': self.pyrace.car.current_check,
            'crash': not self.pyrace.car.is_alive,
        }
        return np.array(obs, dtype=np.float32), reward, done, False, info

    def render(self):
        if self.is_view:
            self.pyrace.view_(self.msgs)

    def set_view(self, flag):
        self.is_view = flag

    def set_msgs(self, msgs):
        self.msgs = msgs

    def close(self):
        pass
