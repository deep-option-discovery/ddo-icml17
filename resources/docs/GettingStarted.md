# Getting Started

This section will provide a basic tutorial of how to use the package with a GridWorld example. The rest of
the documentation will describe more advanced functionality.

## Environments
Environments are stateful object that encode the Markov Decision Process (MDP) structure. The include functionality
for initializing the environment, retrieving observations, and taking actions. All environments must extend segmentcentroid.envs.AbstractEnv.
The class contains the following abstract methods:

```
    """
    This function initializes the environment
    """
    def init(self, state=None, time=0, reward=0):


    """
    This function returns the current state, time, total reward, and termination
    """
    def getState(self):


    """
    This function takes an action
    """
    def play(self, action):


    """
    This function determins the possible actions at a state s, if none provided use 
    current state
    """
    def possibleActions(self, s=None):

```

The details of this will be described in a future section, but one instantiation of AbstractEnviornment is GridWorldEnv. This 
implements these methods in a GridWorld. This is a 2D grid where an agent can move up-down-left-right and its goal is to get to a goal state from a starting point. Gridworlds are constructed with a map, which is a integer grid (0 free space, 1 wall, 2 start, 3, goal, 4 pit).
```
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
1 2 3 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1
1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1
1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1
1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1
1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1
1 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 1
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
```
This map can be loaded to construct a gridworld env using numpy
```
import numpy
from segmentcentroid.envs.GridWorldEnv import GridWorldEnv

gmap = numpy.loadtxt('resources/GridWorldMaps/experiment1.txt', dtype=np.uint8)
g = GridWorldEnv(copy.copy(gmap))       
```

## Planners
To be able to learn hierarchical primitives we need to get demonstration trajectories. The package provides a series of planners to get demonstrations from humans and algorithmic supervisors. All planners take an environment as an argument, and have a plan function. This function returns a trajectory (a list of state, action tuples) with a maximum depth, potentially from a pre-specified starting point.

```
    """
    This function returns a trajectory [(s,a) tuples]
    """
    def plan(self, max_depth, start=None):

```

For GridWorld (and other relatively simple discrete MDPs) we can use value iteration to find a plan. Value Iteration solves a dynamic program to get an optimal policy, and ValueIterationPlanner provides a wrapper for the value iteration algorithm. We can sample trajectories from a gridworld as follows:

```
from segmentcentroid.planners.value_iteration import ValueIterationPlanner

v = ValueIterationPlanner(g)
traj = v.plan(max_depth=100)
g.visualizePlan(traj)
```

The problem is that this gives us a demonstration for a single instance of the problem, if we change the start and the end position then we get additional trajectories:
```
g = GridWorldEnv(copy.copy(gmap), random_start=True)    
v = ValueIterationPlanner(g)
traj = v.plan(max_depth=100)
g.visualizePlan(traj)
```

For the above map, a sample of 50 trajectories looks as follows:
![alt text](https://bytebucket.org/sjyk/segment-centroid/raw/e57d62ffd4f7aef770d8943508436a07c271809a/resources/results/exp1-trajs.png?token=4a75ff20b4340c13f1d81c1038786720a413bcb0 "")


## Models
We would like the learn primitives from the provided plans. The first step is to create a model class this needs to define the state-space, 
action space number of primitives and any additional variables. It must also define two neural networks a representation for the policy
and a representation for the transitions. These can be constructed from segmentcentroid.tfmodels.models.

```
from .TFModel import TFNetworkModel
from .models import *

class GridWorldModel(TFNetworkModel):
    
    """
    This class defines the abstract class for a tensorflow model for the primitives.
    """

    def __init__(self, 
                 k,
                 statedim=(2,1), 
                 actiondim=(4,1), 
                 hidden_layer=32):

        self.hidden_layer = hidden_layer
        
        super(GridWorldModel, self).__init__(statedim, actiondim, k)


    def createPolicyNetwork(self):

        return multiLayerPerceptron(self.statedim[0], 
                                    self.actiondim[0],
                                    self.hidden_layer)

    def createTransitionNetwork(self):

        return multiLayerPerceptron(self.statedim[0], 
                                    2,
                                    self.hidden_layer)


```

Once we create the model, we can create the object (find two primitives):
```
m  = GridWorldModel(2)
```

## Training Models
To train a model, we first must create a tensorflow optimizer, and set the number of iterations:
```
import tensorflow as tf
opt = tf.train.AdamOptimizer(learning_rate=1e-2)
m.train(opt, demonstrations, iterations=100)
```

We can also checkpoint models and restore models:
```
m.checkpoint_file = '/tmp/model.bin'
m.checkpoint_freq = 10

...

m.restore()
```





