"""
This class defines a loader that loads 
from a directory of CSV files of low 
dimensional states
"""

from os import listdir
from os.path import isfile, join
import numpy as np
from .AbstractPlanner import *

class FileLoader(AbstractPlanner):

    def __init__(self, directory, delim=" "):

        #declaring variables that should be set
        self.directory = directory
        self.delim = delim

        super(FileLoader, self).__init__(None)


    """
    This function returns a trajectory [(s,a) tuples]
    """
    def plan(self, max_depth=-1, start=None):
        files = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
        index = np.random.choice(len(files))
        f = open(self.directory+"/"+files[index], "r")
        
        lines = f.readlines()
        states = [np.array([float(li) for li in l.split(self.delim)][self.START:self.END]) for l in lines]

        if max_depth == -1:
            max_depth = len(states)
        else:
            max_depth = min(max_depth, len(states))

        traj = []

        for t in range(1, max_depth):
            xt = states[t-1]
            xtp = states[t]
            traj.append((xt, xtp-xt))

        return traj
