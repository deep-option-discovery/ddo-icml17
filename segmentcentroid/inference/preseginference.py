"""
This class defines the main inference logic
"""

class SegCentroidInferenceDiscrete(object):

    def __init__(self, policy_class, k):
        self.policy_class = policy_class
        self.k = k


    #X is a list of segmented trajectories
    def fit(self, X, statedims, actiondims, max_iters=100, learning_rate=0.001):
        
        #create k initial policies
        policies = [copy.copy(self.policy_class(statedims, actiondims)) for i in range(0,self.k)]


        #initialize q and P for iterations
        q = np.matrix(np.ones((len(X),self.k)))/self.k
        P = np.matrix(np.ones((self.k,1)))/self.k

        
        #Outer Loop For Gradient Descent
        for it in range(0, max_iters):

            #print("Iteration", it, q, P)
            #print(P)

            q, P = self._updateQP(X, policies, q, P)


            for seg in range(0, self.k):
                policies[seg].descent(self._batchGrad(X, policies[seg],seg, q), learning_rate)

        return q, P, policies

            
    """
    Defines the inner loops
    """

    def _batchGrad(self, X, policy, policy_index, q):

        gradSum = None

        m = len(X)

        for plan in range(0, m):

            traj = X[plan]

            pointGrad = q[plan, policy_index]*self._trajLogDeriv(traj, policy)

            if gradSum is None:
                gradSum = pointGrad
            else:
                gradSum = gradSum + pointGrad

        return gradSum*1.0/m



    def _trajLogDeriv(self, traj, policy):
        gradSum = None

        for t in range(0, len(traj)):
            obs = np.matrix(traj[t][0])
            action = traj[t][1]
            deriv = policy.log_deriv(obs, action)

            if gradSum is None:
                gradSum = deriv
            else:
                gradSum = gradSum + deriv

        return gradSum


    def _updateQP(self, X, policies, q, P):

        #how many trajectories are there in X
        m = len(X)

        qorig = copy.copy(q)
        Porig = copy.copy(P)

        #for each trajectory
        for plan in range(0, m):

            #for each segments
            for seg in range(0, self.k):

                q[plan, seg] = P[seg]*self._segLikelihood(X[plan], policies[seg])

            #print q[plan, :], plan, P[seg]
      
        normalization = np.matrix(np.sum(q, axis=1))

        normalization_matrix = np.tile(1/normalization, [1,self.k])
        q = np.multiply(q, normalization_matrix)
        P = np.matrix(np.sum(q, axis=0)).T/m


        #test if nan
        if np.product(1-np.isnan(q)) == 0:
            return qorig, Porig

            
        return q,P



    def _segLikelihood(self, traj, policy):
        product = 1

        for t in range(0, len(traj)):

            obs = np.matrix(traj[t][0])

            pred = np.squeeze(policy.eval(obs))

            action = traj[t][1]
            
            preda = pred[action]

            product = preda * product

        return product


