"""
This class defines a loader that loads 
the JHU Jigsaws dataset
"""
from os import listdir
from os.path import isfile, join
import numpy as np
from .AbstractPlanner import *

class JigsawsPlanner(AbstractPlanner):

    def __init__(self, 
                 kdirectory, 
                 vdirectory=None,
                 gtdirectory=None):

        #declaring variables that should be set
        self.directory = kdirectory
        self.vdirectory = vdirectory
        self.gtdirectory = gtdirectory

        super(JigsawsPlanner, self).__init__(None)


    """
    This function returns a trajectory [(s,a) tuples]
    """
    def plan(self, max_depth=-1, sampling=30, start=None):
        files = [f for f in listdir(self.directory) if isfile(join(self.directory, f))]
        index = np.random.choice(len(files))
        
        f = open(self.directory+"/"+files[index], "r")

        mask = set([38,39,40,56,57,58,59,75])

        lines = f.readlines()
        states = [np.array([float(li) for i, li in enumerate(l.split()) if i in mask]) for l in lines]

        #print(states[0].shape[0])

        """
        #median filter
        for i in range(0, len(states)):
            wsize = 5
            window = range(i, min(i+wsize, len(states)))
            filstate = np.zeros((states[i].shape[0],wsize))
            for j,w in enumerate(window):
                filstate[:,j] = states[w]
            states[i] = np.median(filstate,axis=1)
        """

        #print(states[i].shape[0])


        if self.vdirectory != None:
            demoname = files[index].split('.')[0]
            videoname = self.vdirectory+ "/processed-" + demoname + "_capture1.avi"
            
            import cv2 #only if you want videos

            cap = cv2.VideoCapture(videoname)

            ret = True

            videos = []

            while ret:
                ret, frame = cap.read()
                if ret:
                    videos.append(cv2.resize(frame, (0,0), fx=0.25, fy=0.25) )

            offset = len(videos) - len(states)

            if offset < 0 or videos[0].shape != (120, 160, 3):
                raise ValueError("Dity Data: Misalignment between video and kinematics", videoname)


            if max_depth == -1:
                max_depth = len(states)
            else:
                max_depth = min(max_depth, len(states))

            traj = []

            for t in range(sampling, max_depth, sampling):

                xt = (states[t], videos[t+offset])
                xtp = states[t]

                a = xtp-states[t-sampling]
                print(a)

                traj.append((xt, a))
        else:

            if max_depth == -1:
                max_depth = len(states)
            else:
                max_depth = min(max_depth, len(states))

            traj = []

            for t in range(1, max_depth):
                xt = states[t-1]
                xtp = states[t]
                a = xtp-states[t-1]

                traj.append((xt, a))

        
        f.close()

        if self.gtdirectory != None:
            demoname = files[index].split('.')[0]
            f = open(self.gtdirectory+"/"+files[index], "r")
            lines = [l for l in f.readlines()] #only detect suturing starts

            #print([int(l.strip().split()[1]) for l in lines if 'G14' in l.strip().split()[2] ])

            return traj, [int(l.strip().split()[1]) for l in lines]
        
        else:
            return traj


    def visualizePlans(self, trajs, model, filename=None, ground_truth=None):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        colors = np.random.rand(model.k,3)

        for i, traj in enumerate(trajs):
            Q = np.argmax(model.fb.fit([traj])[0][0], axis=1)
            segset = { j : np.where(Q == j)[0] for j in range(model.k) }

            for s in segset:

                plt.scatter(segset[s], segset[s]*0 + i, color=colors[s],marker='.')

            if ground_truth != None:
                plt.scatter(ground_truth[i], np.ones((len(ground_truth[i]),1))*i, marker='x')

        if filename == None:
            plt.show()
        else:
            plt.savefig(filename)
            




        

