
from .AbstractPlanner import *

"""
This class defines a general value iteration technique in
discrete state-spaces and action spaces.
"""

class ValueIteration(object):

    def __init__(self, state_list, action_list, dynamics_model, reward_function):
        """
        Pass in an iterable of states, actions, dynamics_model, reward_function

        Positional arguments:
        state_list -- list of all feasible states
        action_list -- list of all feasible actions
        dynamics_model -- map from (state,action) to a list of (state, prob) tuples
        """
        self.state_list = state_list
        self.action_list = action_list
        self.value_function = dict([(s,0) for s in state_list])
        self.dynamics_model = dynamics_model
        self.reward_function = reward_function
        self.policy = dict([(s,0) for s in state_list])
        self.fitted = False


    def fit(self, horizon=100):
        """
        Runs value iteration.

        Keyword arguments:
        horizon -- int number of iterations
        """

        for t in range(horizon):
            self._one_step_bb()


        self.fitted =True

        return self.policy, self.value_function

    def _one_step_bb(self):
        """
        Performs one step of value iteration
        """

        for state in self.value_function:
            vupdate, pupdate = self._argmaxa(state)
            self.value_function[state] = vupdate
            self.policy[state] = pupdate

            
    def _expectation(self, state, action):
        """
        Cacluates the expected value at state, action
        """

        total = 0
        for state_prob in self.dynamics_model[(state,action)]:
            
            sp = state_prob[0]
            p = state_prob[1]

            total = total + p*(self.reward_function(state, action) \
                          + self.value_function[sp])

        return total

    def _argmaxa(self, state):
        """
        Finds the action that maximizes value
        """

        maxv = 0
        maxa = None

        for action in self.action_list:
                if (state,action) in self.dynamics_model:
                    newv = self._expectation(state,action)

                    #init if none
                    if maxa == None:
                        maxv = newv
                        maxa = action

                    #or if new action is better
                    if maxv < newv:
                        maxv = newv
                        maxa = action

                    #print(newv, action)
        return maxv, maxa


"""
Wraps the value iteration routine into a 
planner.
"""

class ValueIterationPlanner(AbstractPlanner):

    #have to pass in an abstractenv descendent
    def __init__(self, domain, horizon=100):

        v = ValueIteration(domain.getAllStates(), 
                           domain.getAllActions(), 
                           domain.getDynamicsModel(), 
                           domain.getRewardFunction())

        self.policy, self.value = v.fit(horizon=horizon)

        super(ValueIterationPlanner, self).__init__(domain)


    def plan(self, max_depth, start=None):

        traj = []

        self.domain.init(state=start)

        for t in range(max_depth):
            state, time, reward, termination = self.domain.getState()

            if termination:
                return traj

            action = self.policy[tuple(state)]
            self.domain.play(action)
            traj.append((state, action))

        return traj











