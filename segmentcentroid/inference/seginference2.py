
import numpy as np
import copy
from sklearn.preprocessing import normalize

class JointSegCentroidInferenceDiscrete(object):

    def __init__(self, 
                 policy_class, 
                 transition_class, 
                 k, 
                 statedims, 
                 actiondims):

        self.policy_class = policy_class
        self.transition_class = transition_class
        self.k = k

        #initialize parameters
        self.policies = [copy.copy(self.policy_class(statedims, actiondims)) for i in range(0,self.k)]
        self.transitions = [copy.copy(self.transition_class(statedims, 1)) for i in range(0,self.k)]

        self.X = None
        self.Q = None
        self.B = None
        self.P = np.ones((self.k, self.k))/self.k

        self.iter_state = {}


    #Xl is a list of trajectories
    def fit(self, 
            Xl, 
            max_iters=20, 
            max_liters=1, 
            learning_rate=0.01):

        for it in range(0, max_iters):
            for i, traj in enumerate(Xl):
                self.init_iter(i, traj)
                self.iter_state[i] = self.fitTraj(traj, max_iters=max_liters, learning_rate=learning_rate)
                print("Iteration ", it,": ",i, np.argmax(self.Q, axis=1), np.argmax(self.B, axis=1))

        return self.policies, self.transitions

    def init_iter(self,index, X):
        if index in self.iter_state:
            print("loaded")
            self.Q, self.B, self.fq, self.bq, self.P = self.iter_state[index]
        else:
            #uniform probability of each segment
            self.Q = np.ones((len(X), self.k))/self.k
            self.fq = np.ones((len(X)+1, self.k))/self.k
            self.bq = np.ones((len(X)+1, self.k))/self.k

            #uniform probability of next timestep terminating
            self.B = np.ones((len(X), self.k))/2

            #uniform transition probabilities
            self.P = np.ones((self.k, self.k))/self.k


    #X is a trajectory
    def fitTraj(self, X, max_iters=50, learning_rate=0.01):
        #all the data
        self.X = X

        for it in range(0, max_iters):

            print("Sub-Iteration ", it, np.argmax(self.Q, axis=1), np.argmax(self.B, axis=1))

            self.forward()

            self.backward()

            self.Q = np.multiply(self.fq,self.bq)

            self.Q = normalize(self.Q, norm='l1', axis=1)

            for t in range(len(self.X)-1):
                self.B[t,:] = self.termination(t)

            for seg in range(0, self.k):

                if self.policies[0].isTabular:
                    self.policies[seg].descent(self._tabGradP(X, seg, self.Q), learning_rate)
                else:
                    self.policies[seg].descent(self._batchGradP(X, seg, self.Q), learning_rate)

                if self.transitions[0].isTabular:
                    self.transitions[seg].descent(self._tabGradT(X, seg, self.B), learning_rate)
                else:
                    self.transitions[seg].descent(self._batchGradT(X, seg, self.B), learning_rate)

        return self.Q, self.B, self.fq, self.bq, self.P


    def forward(self, t=None):

        if t == None:
            t =len(self.X)-1

        #initialize table
        forward_dict = {}
        for h in range(self.k):
            forward_dict[(0,h)] = 1

        #dynamic program
        for cur_time in range(t):

            state = self.X[cur_time][0]
            next_state = self.X[cur_time+1][0]
            action = self.X[cur_time][1]

            for hp in range(self.k):
                
                total = 0


                for h in range(self.k):
                    total = total + forward_dict[(cur_time, h)]*self._pi_a_giv_s(state,action,self.policies[h])*\
                                                               (self.P[hp,h]*self._pi_term_giv_s(next_state,self.transitions[h]) + \
                                                                 (1-self._pi_term_giv_s(next_state,self.transitions[h]))*(hp == h)
                                                                )
                forward_dict[(cur_time+1, hp)] = total

        for k in forward_dict:
            #print(forward_dict[k], k)
            self.fq[k[0],k[1]] = max(forward_dict[k],1e-6)


    def backward(self, t=None):

        if t == None:
            t = 0

        #initialize table
        backward_dict = {}
        for h in range(self.k):
            backward_dict[(len(self.X),h)] = 1

        rt = np.arange(len(self.X), t, -1)

        #dynamic program
        for cur_time in rt:

            state = self.X[cur_time-1][0]
            
            if cur_time == len(self.X):
                next_state = state
            else:
                next_state = self.X[cur_time][0]

            action = self.X[cur_time-1][1]

            for h in range(self.k):
                
                total = 0


                for hp in range(self.k):
                    total = total + backward_dict[(cur_time, hp)]*\
                                                               (self.P[hp,h]*self._pi_term_giv_s(next_state,self.transitions[h]) + \
                                                                 (1-self._pi_term_giv_s(state,self.transitions[h]))*(hp == h)
                                                                )
                backward_dict[(cur_time-1, h)] = total*self._pi_a_giv_s(state,action,self.policies[h])


                #print(backward_dict[(cur_time-1, hp)], total, cur_time, len(self.X))

        for k in backward_dict:
            self.bq[k[0],k[1]] = max(backward_dict[k],1e-6)


    def termination(self, t):

        state = self.X[t][0]
            
        if t+1 == len(self.X):
            next_state = state
        else:
            next_state = self.X[t+1][0]

        action = self.X[t][1]


        termination = {}

        for h in range(self.k):
            
            total = 0

            for hp in range(self.k):

                #print("h",h)
                total = total + self.fq[t,h]*\
                            self._pi_a_giv_s(state,action,self.policies[h])*\
                            self.P[hp,h]*\
                            self._pi_term_giv_s(next_state,self.transitions[h])*\
                            self.bq[t+1,hp]

            termination[h] = total

        return [termination[h] for h in range(self.k)]


    def _pi_a_giv_s(self, s,a, policy):
        obs = np.matrix(s)
        pred = np.squeeze(policy.eval(obs))
        action = a
        #print(pred)
        return pred[action]


    def _pi_term_giv_s(self, s, transition):
        obs = np.matrix(s)
        pred = np.squeeze(transition.eval(obs))
        return pred



    #gradient routines, one for differentiable
    #one for tabular

    def _batchGradP(self, X, policy_index, q):

        gradSum = None

        m = len(X)-1

        q = normalize(q, norm='l1', axis=1)

        for t in range(0, m):

            obs = np.matrix(X[t][0])
            action = X[t][1]

            pointGrad = q[t, policy_index]*self.policies[policy_index].log_deriv(obs, action)

            if gradSum is None:
                gradSum = pointGrad
            else:
                gradSum = gradSum + pointGrad

        return gradSum*1.0/m


    def _batchGradT(self, X, policy_index, b):

        gradSum = None

        m = len(X)-1

        #b = normalize(b, norm='l1', axis=1)

        for t in range(0, m):

            obs = np.matrix(X[t+1][0])

            pointGrad1 = b[t, policy_index]*self.transitions[policy_index].log_deriv(obs, 1)

            pointGrad2 = (1-b[t, policy_index])*self.transitions[policy_index].log_deriv(obs, 0)

            #print(pointGrad)

            if gradSum is None:
                gradSum = pointGrad1+ pointGrad2
            else:
                gradSum = gradSum + pointGrad1 + pointGrad2

        return gradSum*1.0/m


    def _tabGradP(self, X, policy_index, q):

        gradSum = []

        m = len(X)-1

        #print(self.fq, self.bq)
        q = normalize(q, norm='l1', axis=1)
       

        for t in range(0, m):

            obs = np.matrix(X[t][0])
            action = X[t][1]

            #print(q[t, policy_index], policy_index)

            pointGrad = (q[t, policy_index], self.policies[policy_index].log_deriv(obs, action))

            gradSum.append(pointGrad)

        return gradSum


    def _tabGradT(self, X, policy_index, b):

        gradSum = []

        b = normalize(b, norm='l1', axis=1)

        m = len(X)-1

        for t in range(0, m):

            obs = np.matrix(X[t+1][0])

            pointGrad1 = (b[t, policy_index], \
                         self.transitions[policy_index].log_deriv(obs, 1))

            pointGrad2 = ((1-b[t, policy_index]), \
                         self.transitions[policy_index].log_deriv(obs, 0))

            gradSum.append(pointGrad1)
            gradSum.append(pointGrad2)

        return gradSum










        
