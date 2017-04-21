"""
Implements some basic utils for manipulating trajectories
"""

def waypoint_segment(traj, waypoints):
    segments = {}
    
    cur_index = 0
    segments[cur_index] = []

    for t in traj:

        segments[cur_index].append(t)
        
        if cur_index < len(waypoints) and \
           tuple(t[0]) == waypoints[cur_index]:

            cur_index = cur_index + 1
            segments[cur_index] = []

    return [segments[k] for k in segments]



