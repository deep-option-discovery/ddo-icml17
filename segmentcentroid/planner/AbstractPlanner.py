"""
This class defines an abstract planner,
all planners derive from this class
"""

class AbstractPlanner(object):

    def __init__(self, domain):

        #declaring variables that should be set

        self.domain = domain


    """
    This function returns a trajectory [(s,a) tuples]
    """
    def plan(self, max_depth, start=None):
        raise NotImplemented("Must implement a plan command")

        

     
