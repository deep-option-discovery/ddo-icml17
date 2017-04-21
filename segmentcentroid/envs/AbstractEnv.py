"""
This class defines an abstract environment,
all environments derive from this class
"""

class AbstractEnv(object):

    def __init__(self):

        #declaring variables that should be set

        self.state = None

        self.termination= False

        self.time = 0

        self.reward = 0


    """
    This function initializes the envioronment
    """
    def init(self, state=None, time=0, reward=0):
        raise NotImplemented("Must implement an init command")


    """
    This function returns the current state, time, total reward, and termination
    """
    def getState(self):
        return self.state, self.time, self.reward, self.termination


    """
    This function takes an action
    """
    def play(self, action):
        raise NotImplemented("Must implement a play command")

    """
    This function determins the possible actions at a state s, if none provided use 
    current state
    """
    def possibleActions(self, s=None):
        raise NotImplemented("Must implement a play command")


    """
    This function rolls out a policy which is a map from state to action
    """
    def rollout(self, policy):
        trajectory = []

        while not self.terminated:
            self.play(policy(self.state))
            trajectory.append(self.getState())

        return trajectory


        

     
