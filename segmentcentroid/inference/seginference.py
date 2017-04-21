
import numpy as np
import copy
from sklearn.preprocessing import normalize

class JointSegCentroidInferenceDiscrete(object):

    def __init__(self, policy_class, transition_class, k):
        self.policy_class = policy_class
        self.transition_class = transition_class
        self.k = k


    #X is a trajectory
    def fit(self, X, statedims, actiondims, max_iters=20, learning_rate=0.01):
        
        #create k initial policies
        policies = [copy.copy(self.policy_class(statedims, actiondims, unnormalized=True)) for i in range(0,self.k)]

        #create k initial transitions
        transitions = [copy.copy(self.transition_class(statedims, 1, unnormalized=True)) for i in range(0,self.k)]

        #hint transition matrix, defaulted to uniform
        Ph = np.matrix(np.ones((self.k,self.k)))/self.k

        q = np.matrix(np.ones((len(X),self.k)))/self.k
        psi = np.matrix(np.ones((len(X),self.k)))/self.k
       
        #print(Ph)
        #print(self._forward(20, 1, X, policies, transitions, Ph))

        for it in range(0, max_iters):

            self.forward_memo = {}
            self.backward_memo = {}

            q = self._updateQ(q, X, policies, transitions, Ph)
            psi = self._updatePsi(psi, X, policies, transitions, Ph)

            print("Iteration", it, np.argmax(q, axis=1))

            for seg in range(0, self.k):
                policies[seg].descent(self._batchGrad(X, policies[seg],seg, q), learning_rate)
                transitions[seg].descent(self._batchGrad(X, transitions[seg],seg, psi), learning_rate)

        return q, psi, policies, transitions


    def _updateQ(self, q, X, policies, transitions, Ph):
        
        newq = copy.copy(q)

        for h in range(self.k):
            for t in range(len(X)-1):

                if (t,h) in self.forward_memo:
                    forward = self.forward_memo[(t,h)]
                else:
                    forward = self._forward(t, h, X, policies, transitions, Ph)
                    self.forward_memo[(t,h)] = forward

                if (t,h) in self.backward_memo:
                    backward = self.backward_memo[(t,h)]
                else:
                    backward = self._backward(t, h, X, policies, transitions, Ph)
                    self.backward_memo[(t,h)] = backward

                newq[t,h] = forward*\
                            backward

        newq = normalize(newq, axis=1, norm='l1')

        return newq


    def _updatePsi(self, psi, X, policies, transitions, Ph ):

        newpsi = copy.copy(psi)

        for t in range(len(X)-1):
            for h in range(self.k):
                
                total = 0

                for hp in range(self.k):

                    if (t,hp) in self.forward_memo:
                        forward = self.forward_memo[(t,hp)]
                    else:
                        forward = self._forward(t, hp, X, policies, transitions, Ph)
                        self.forward_memo[(t,hp)] = forward

                    total = total + \
                            self._backwardTerm(t,h,X, policies, transitions, Ph)* \
                            Ph[hp, h] * \
                            forward

                    #print(self._backwardTerm(t,h,X, policies, transitions, Ph))

                newpsi[t,h] = total

        newpsi = normalize(newpsi, axis=1, norm='l1')

        print(newpsi)

        return newpsi


    def _backwardTerm(self, t, h, X, policies, transitions, Ph):

        s = X[t][0]
        sp = X[t+1][0]
        a = X[t][1]

        #print(self._backward(t, h, X, policies, transitions, Ph),self._pi_a_giv_s(s,a, policies[h]), self._pi_term_giv_s(sp, policies[h]))

        if (t,h) in self.backward_memo:
            back = self.backward_memo[(t,h)]
        else:
            back = self._backward(t, h, X, policies, transitions, Ph)
            self.backward_memo[(t,h)] = back


        return back*\
               self._pi_a_giv_s(s,a, policies[h])*\
               self._pi_term_giv_s(sp, transitions[h])



    def _backward(self, 
                  #args
                  t,
                  hp,
                  #control state
                  traj, 
                  policies,
                  transitions,
                  Ph):

        state = traj[t][0]
        next_state = traj[t+1][0]
        action = traj[t][1]

        if (t,hp) in self.backward_memo:
            return self.backward_memo[(t,hp)]

        elif t == 0:
            
            rtn = np.sum([ self._pi_a_giv_s(state,action, policies[h])* \
                            (self._pi_term_giv_s(next_state, transitions[h])* \
                            Ph[hp,h])\
                           for h in range(self.k)\
                         ])

            self.backward_memo[(t,hp)] = rtn
            
            return rtn


        else:
            rtn = np.sum([ self._backward(t-1, h, traj, policies, transitions, Ph)* \
                            self._pi_a_giv_s(state,action, policies[h])* \
                            (self._pi_term_giv_s(next_state, transitions[h])* \
                            Ph[hp,h])\
                           for h in range(self.k)\
                         ])
            self.backward_memo[(t,hp)] = rtn
            return rtn

    def _forward(self, 
                  #args
                  t,
                  h,
                  #control state
                  traj, 
                  policies,
                  transitions,
                  Ph):
        
        state = traj[t][0]
        next_state = traj[t+1][0]
        action = traj[t][1]


        if (t,h) in self.forward_memo:
            return self.forward_memo[(t,h)]

        elif t+2 >= len(traj):
            
            rtn = self._pi_a_giv_s(next_state,action, policies[h])
            self.forward_memo[(t,h)] = rtn

            return rtn

        else:
            rtn =  self._pi_a_giv_s(next_state,action, policies[h]) * \
                   np.sum([ self._pi_term_giv_s(state, transitions[h])* \
                            Ph[hp,h]* \
                            self._forward(t+1,hp, traj, policies, transitions, Ph) \
                            for hp in range(self.k)])
            
            self.forward_memo[(t,h)] = rtn
            return rtn



    def _pi_a_giv_s(self, s,a, policy):
        obs = np.matrix(s)
        pred = np.squeeze(policy.eval(obs))
        action = a
        return pred[action]


    def _pi_term_giv_s(self, s, transition):
        obs = np.matrix(s)
        pred = np.squeeze(transition.eval(obs))
        return pred



    def _batchGrad(self, X, policy, policy_index, q):

        gradSum = None

        m = len(X)

        for t in range(0, m):

            obs = np.matrix(X[t][0])
            action = X[t][1]

            pointGrad = q[t, policy_index]*policy.log_deriv(obs, action)

            if gradSum is None:
                gradSum = pointGrad
            else:
                gradSum = gradSum + pointGrad

        return gradSum*1.0/m







        
