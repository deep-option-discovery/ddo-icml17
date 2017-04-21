
from .AbstractEnv import *
import numpy as np
import copy
from matplotlib import colors
import matplotlib.pyplot as plt
import random


"""
This class defines an abstract environment,
all environments derive from this class
"""

class GridWorldGasEnv(AbstractEnv):


    ##All of the constant variables

    # Constants in the map
    EMPTY, BLOCKED, START, GOAL, PIT, AGENT, GAS = range(7) #codes

    ACTIONS = np.array([[-1, 0, -1], [+1, 0, -1], [0, -1, -1], [0, +1, -1]])
    actions_num = 4
    GOAL_REWARD = +1
    PIT_REWARD = -1
    STEP_REWARD = -.001

    #takes in a 2d integer map coded by the first line of comments
    def __init__(self, gmap, noise=0.1, random_start=True, gas_limit=20):

        self.map = gmap

        self.start_state = np.ndarray(3)
        self.start_state[:2] = np.argwhere(self.map == self.START)[0]
        self.start_state[2] = gas_limit
        self.gas_limit = gas_limit

        
        self.ROWS, self.COLS = np.shape(self.map)
        self.statespace_limits = np.array(
            [[0, self.ROWS - 1], [0, self.COLS - 1]])

        self.NOISE = noise

        #if random_start:
        #    self.generateRandomStartGoal()

        #print(self.start_state, np.argwhere(self.map == self.GOAL)[0])

        super(GridWorldGasEnv, self).__init__()


    def generateRandomStartGoal(self, pstart=None, pgoal=None):
        start = np.argwhere(self.map == self.START)[0]
        goal = np.argwhere(self.map == self.GOAL)[0]
        self.map[start[0], start[1]] = self.EMPTY
        self.map[goal[0], goal[1]] = self.EMPTY

        empty_cells = np.argwhere(self.map == self.EMPTY)
        p,_ = np.shape(empty_cells)

        nstart = empty_cells[np.random.choice(np.arange(p)),:]

        if pstart==None:
            nstart = empty_cells[np.random.choice(np.arange(p)),:]
        else:
            nstart = pstart

        if pgoal==None:
            ngoal = empty_cells[np.random.choice(np.arange(p)),:]
        else:
            ngoal = pgoal


        if (nstart[0] == ngoal[0] and nstart[1] == ngoal[1]) \
            and (pgoal == None or pstart == None):
            self.map[start[0], start[1]] = self.START
            self.map[goal[0], goal[1]] = self.GOAL
            self.generateRandomStartGoal(pstart,pgoal)
        else:
            self.map[nstart[0], nstart[1]] =self.START
            self.map[ngoal[0], ngoal[1]] =self.GOAL
            self.start_state = np.ndarray(3)
            self.start_state[:2] = np.argwhere(self.map == self.START)[0]
            self.start_state[2] = self.gas_limit


    #helper method returns the terminal state
    def isTerminal(self, s=None):

        if s is None:
            s = self.state

        if self.map[s[0], s[1]] == self.GOAL:
            return True

        if self.map[s[0], s[1]] == self.PIT:
            return True

        if s[2] <= 0:
            return True
            
        return False


    """
    Determines the possible actions at a state
    """
    def possibleActions(self, s=None):
        if s is None:
            s = self.state
        possibleA = np.array([], np.uint8)

        if s[2] == 0:
            return possibleA

        for a in range(self.actions_num):
            ns = s + self.ACTIONS[a]
            if (
                    ns[0] < 0 or ns[0] == self.ROWS or
                    ns[1] < 0 or ns[1] == self.COLS or
                    self.map[int(ns[0]), int(ns[1])] == self.BLOCKED):
                continue
            possibleA = np.append(possibleA, [a])
        return possibleA


    """
    This function initializes the envioronment
    """
    def init(self, state=None, time=0, reward=0 ):
        if state == None:
            self.state = self.start_state.copy()
        else:
            self.state = state

        self.start_state[2] = self.gas_limit
        self.time = time
        self.reward = reward
        self.termination = self.isTerminal()


    """
    This function returns the current state, time, total reward, and termination
    """
    def getState(self):
        return self.state, self.time, self.reward, self.termination


    """
    This function takes an action
    """
    def play(self, a):


        #throws an error if you are stupid
        if a not in self.possibleActions():
            raise ValueError("Invalid Action!!")

        #copies states to make sure no concurrency issues
        r = self.STEP_REWARD
        ns = self.state.copy()

        if np.random.rand(1,1) < self.NOISE:
            # Random Move
            a = np.random.choice(self.possibleActions())

        # Take action
        ns = self.state + self.ACTIONS[a]

        # Check bounds on state values
        if (ns[0] < 0 or ns[0] == self.ROWS or
            ns[1] < 0 or ns[1] == self.COLS or
            self.map[ns[0], ns[1]] == self.BLOCKED):
            ns = self.state.copy()
        else:
            # If in bounds, update the current state
            self.state = ns.copy()

        # Compute the reward
        if self.map[ns[0], ns[1]] == self.GOAL:
            r = self.GOAL_REWARD

        if self.map[ns[0], ns[1]] == self.PIT:
            r = self.PIT_REWARD

        if self.map[ns[0], ns[1]] == self.GAS:
            ns[2] = self.gas_limit

        self.state = ns
        self.time = self.time + 1
        self.reward = self.reward + r
        self.termination = self.isTerminal()

    """
    This function rolls out a policy which is a map from state to action
    """
    def rollout(self, policy):
        trajectory = []

        while not self.terminated:
            self.play(policy(self.state))
            trajectory.append(self.getState())

        return trajectory



    ##plannable environment

    def getRewardFunction(self):

        def _reward(ns,a):
            # Compute the reward
            r = 0
            if self.map[ns[0], ns[1]] == self.GOAL:
                r = self.GOAL_REWARD
            if self.map[ns[0], ns[1]] == self.PIT:
                r = self.PIT_REWARD
            return r

        return _reward


    def getAllStates(self):
        state_limits = np.shape(self.map)
        return [(i,j,k) for i in range(0, state_limits[0]) for j in range(0, state_limits[1]) for k in np.arange(0,self.gas_limit+1)] 

    def getAllActions(self):
        return range(0,4)

    def getDynamicsModel(self):

        dynamics = {}
        states = self.getAllStates()

        for s in states:
            possibleActions = self.possibleActions(s)

            for a in possibleActions:
                dynamics[(s,a)] = []
                expected_step = (s[0] + self.ACTIONS[a][0], s[1] + self.ACTIONS[a][1], s[2] + self.ACTIONS[a][2])

                if self.map[s[0]+ self.ACTIONS[a][0],s[1]+ self.ACTIONS[a][1]] == self.GAS:
                    expected_step = (s[0] + self.ACTIONS[a][0], s[1] + self.ACTIONS[a][1], self.gas_limit)


                dynamics[(s,a)].append( (expected_step, 1-self.NOISE))

                for ap in possibleActions:
                    if ap != a:
                        expected_step = (s[0] + self.ACTIONS[ap][0], s[1] + self.ACTIONS[ap][1], s[2] + self.ACTIONS[ap][2])
                        dynamics[(s,a)].append( (expected_step, self.NOISE/(len(possibleActions)-1)))                        

        return dynamics




    ###visualization routines
    def visualizePolicy(self, policy, transitions=None, blank=False, filename=None):
        cmap = colors.ListedColormap(['w', '.75', 'b', 'g', 'r', 'k'], 'GridWorld')

        plt.figure()

        newmap = copy.copy(self.map)

        if blank:
            start = np.argwhere(self.map == self.START)[0]
            goal = np.argwhere(self.map == self.GOAL)[0]
            newmap[start[0], start[1]] = self.EMPTY
            newmap[goal[0], goal[1]] = self.EMPTY

        #show gw
        plt.imshow(newmap, 
                   cmap=cmap, 
                   interpolation='nearest',
                   vmin=0,
                   vmax=4)

        ax = plt.axes()

        #show policy
        for state in policy:
            
            if policy[state] == None or \
               self.map[state[0], state[1]] == self.BLOCKED:
                continue

            action = self.ACTIONS[policy[state]]

            alpha = transitions[state]

            dx = action[0]*0.5
            dy = action[1]*0.5

            ax.arrow(state[1], state[0], dy, dx, head_width=0.1, fc=(alpha,0,0), ec=(alpha,0,0))


        if filename == None:
            plt.show()
        else:
            plt.savefig(filename)


    def visualizePlan(self, plan, blank=False, filename=None):
        cmap = colors.ListedColormap(['w', '.75', 'b', 'g', 'r', 'k'], 'GridWorld')

        plt.figure()

        newmap = copy.copy(self.map)

        if blank:
            start = np.argwhere(self.map == self.START)[0]
            goal = np.argwhere(self.map == self.GOAL)[0]
            newmap[start[0], start[1]] = self.EMPTY
            newmap[goal[0], goal[1]] = self.EMPTY

        #show gw
        plt.imshow(newmap, 
                   cmap=cmap, 
                   interpolation='nearest',
                   vmin=0,
                   vmax=4)

        ax = plt.axes()

        c = (1,0,0,0.3)
        #show policy
        for sa in plan:
            
            state = sa[0]
            actioni = np.argwhere(np.ravel(sa[1])>0)[0]

            if self.map[state[0], state[1]] == self.BLOCKED:
                continue

            action = self.ACTIONS[actioni]
            dx = action[0,0]*0.5
            dy = action[0,1]*0.5
            ax.arrow(state[1], state[0], dy, dx, head_width=0.1, fc=c, ec=c)

        if filename == None:
            plt.show()
        else:
            plt.savefig(filename)





        

     
