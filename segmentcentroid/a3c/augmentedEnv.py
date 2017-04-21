import gym
from gym import error, spaces, utils
from gym.utils import seeding
import cv2
import numpy as np
import ray
from segmentcentroid.tfmodel.AtariVisionModel import AtariVisionModel
import tensorflow as tf

class AugmentedEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, gymEnvName, model_weights, k):

    model = AtariVisionModel(k)
    model.sess.run(tf.initialize_all_variables())
    variables = ray.experimental.TensorFlowVariables(model.loss, model.sess)
    
    variables.set_weights(model_weights)

    self.model = model
    self.env = gym.make(gymEnvName)
    self.action_space = spaces.Discrete(self.env.action_space.n + model.k) 
    self.obs = None
    self.done = False

  def _process_frame42(self, frame):
    frame = frame[34:(34+160), :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    frame = frame.mean(2)
    frame = frame.astype(np.float32)
    frame *= (1.0 / 255.0)
    frame = np.reshape(frame, [42, 42, 1])
    return frame

  def _step(self, action):
    N = self.env.action_space.n
    actions = np.eye(N)
    
    if action < N:
        output = self.env._step(action)
        self.obs, _, self.done , _ = output
    else:
        obs = self.obs
        proc_obs = self._process_frame42(obs)
        done = self.done
        term = 0 

        while (not done) and (np.random.rand(1) > term):  
            l = [ np.ravel(self.model.evalpi(action-N, [(proc_obs, actions[j,:])] ))  for j in range(N)]
            output = self._step(np.argmax(l))
            obs = self.obs
            done = self.done
            term = np.ravel(self.model.evalpsi(action-N, [(proc_obs, actions[1,:])]))

    return output

  def _reset(self):
    self.obs = self.env._reset()
    self.done = False
    return self.obs

  def _render(self, mode='human', close=False):
    return self.env._render(mode, close)