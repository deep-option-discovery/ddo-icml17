
import numpy as np


class segclust(object):
    def __init__(self, model_class):
        self.model_class = model_class

    def calc_alpha(self, traj, model):
        N = len(traj)
        K = self.model_class.K
        alpha = []
        for i in range(N):
            traj_i = traj[i]
            T = len(traj_i) - 1
            alpha_i = np.zeros((T, K))
            for h in range(K):
                alpha_i[0, h] = model.P.calc_log(None, h)
            for t in range(T-1):
                s = traj_i[t].s
                a = traj_i[t].a
                s_ = traj_i[t+1].s
                for h_ in range(K):
                    alpha_i[t+1, h_] = logsumexp(
                        [alpha_i[t, h] + model.pi[h].calc_log(s, a) +
                         log(model.psi[h].calc(s_) * (model.P.calc(h, h_) - (h == h_)) + (h == h_))
                         for h in range(K)])
            alpha.append(alpha_i)
        return alpha

    def calc_beta(self, traj, model):
        N = len(traj)
        K = self.model_class.K
        beta = []
        for i in range(N):
            traj_i = traj[i]
            T = len(traj_i) - 1
            beta_i = np.zeros((T, K))
            for h in range(K):
                beta_i[T-1, h] = (model.pi[h].calc_log(traj_i[T-1].s, traj_i[T-1].a) +
                                  model.psi[h].calc_log(traj_i[T].s) +
                                  model.P.calc_log(h, None))
            for t in range(T-2, -1, -1):
                s = traj_i[t].s
                a = traj_i[t].a
                s_ = traj_i[t+1].s
                for h in range(K):
                    beta_i[t, h] = model.pi[h].calc_log(s, a) + logsumexp(
                        [log(model.psi[h].calc(s_) * (model.P.calc(h, h_) - (h == h_)) + (h == h_)) +
                         beta_i[t+1, h_]
                         for h_ in range(K)])
            beta.append(beta_i)
        return beta

    @staticmethod
    def calc_q(alpha, beta):
        N = len(alpha)
        q = []
        for i in range(N):
            T, K = alpha[i].shape
            q_i = np.zeros((T, K))
            for t in range(T):
                for h in range(K):
                    q_i[t, h] = alpha[i][t, h] + beta[i][t, h]
                q_i[t] = normalized_exp(q_i[t])
            q.append(q_i)
        return q

    @staticmethod
    def calc_b(traj, model, alpha, beta):
        N = len(alpha)
        b = []
        for i in range(N):
            traj_i = traj[i]
            alpha_i = alpha[i]
            beta_i = beta[i]
            T, K = alpha_i.shape
            b_i = np.zeros((T, K))
            for h in range(K):
                b_i[T-1, h] = alpha_i[T-1, h] + beta_i[T-1, h]
            b_i[T-1] = normalized_exp(b_i[T-1])
            for t in range(T-1):
                s = traj_i[t].s
                a = traj_i[t].a
                s_ = traj_i[t+1].s
                for h in range(K):
                    b_i[t, h] = alpha_i[t, h] + model.pi[h].calc_log(s, a) + model.psi[h].calc_log(s_) + logsumexp(
                        [model.P.calc_log(h, h_) + beta_i[t+1, h_] for h_ in range(K)])
                b_i[t] = normalized_exp(b_i[t])
            b.append(b_i)
        return b

    @staticmethod
    def calc_grad(traj, model, q, b):
        grad = np.zeros(model.n_params)
        N = len(traj)
        for i in range(N):
            traj_i = traj[i]
            q_i = q[i]
            b_i = b[i]
            T, K = q_i.shape
            for t in range(T):
                s = traj_i[t].s
                a = traj_i[t].a
                s_ = traj_i[t+1].s
                for h in range(K):
                    grad += q_i[t, h] * model.pi[h].calc_grad_log(s, a)
                    grad += b_i[t, h] * model.psi[h].calc_grad_log(s_)
        return grad
