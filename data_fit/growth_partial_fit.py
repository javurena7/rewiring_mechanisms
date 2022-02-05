import numpy as np
import sys
sys.path.append('..')
import asikainen_model as am
import sympy as sym
from scipy import optimize
from scipy.special import expit

def get_dataset(sa, sb, na, c, N, m=10):
    n_iter = 0
    p, t, (P, counts), rho, converg_d = am.run_growing(N, na, c, [sa, sb], track_steps=1, m=m, n_iter=n_iter, p0=[], ret_counts=True)
    Paa, Pbb = [], []
    for laa, lab, lbb in zip(P['l_aa'], P['l_ab'], P['l_bb']):
        if laa + lab + lbb > 0:
            Paa.append(laa/(laa + lab + lbb))
            Pbb.append(lbb/(laa + lab + lbb))
        else:
            Paa.append(0)
            Pbb.append(0)
    #Paa = Paa[int(N/2):]
    #Pbb = Pbb[int(N/2):]
    #counts = counts[int(N/2):]
    obs_counts = np.array(counts)[:, [0, 1, 3, 4]]
    return Paa, Pbb, obs_counts

def get_dataset_EM(sa, sb, na, c, N, m=10):
    n_iter = 0
    p, t, (P, counts), rho, converg_d = am.run_growing(N, na, c, [sa, sb], track_steps=1, m=m, n_iter=n_iter, p0=[], ret_counts=True)
    Paa, Pbb = [], []
    for laa, lab, lbb in zip(P['l_aa'], P['l_ab'], P['l_bb']):
        if laa + lab + lbb > 0:
            Paa.append(laa/(laa + lab + lbb))
            Pbb.append(lbb/(laa + lab + lbb))
        else:
            Paa.append(0)
            Pbb.append(0)
    #Paa = Paa[int(N/2):]
    #Pbb = Pbb[int(N/2):]
    #counts = counts[int(N/2):]
    obs_counts = np.array(counts)[:, [0, 1, 3, 4]]
    return Paa, Pbb, obs_counts, counts

def get_dataset_varm(sa, sb, na, c, N, m_dist=['poisson', 50], m0=10, em=True):
    Paa, Pbb, counts = am.run_growing_varying_m(N, na, c, [sa, sb], m_dist, m0=m0, ret_counts=True)
    if em:
        obs_counts = np.array(counts)[:, [0, 1, 3, 4]]
        return Paa, Pbb, obs_counts, counts
    else:
        return Paa, Pbb, counts

def _lik_catprobs(Paa, Pbb, na, sa, sb, c):
    nb = 1 - na
    Maa = (c/2*(1+Paa-Pbb) + (1-c)*na)*sa
    Mab = (c/2*(1+Pbb-Paa) + (1-c)*nb)*(1-sa)
    Man = Maa + Mab

    Mba = (c/2*(1+Paa-Pbb) + (1-c)*na)*(1-sb)
    Mbb = (c/2*(1+Pbb-Paa) + (1-c)*nb)*sb
    Mbn = Maa + Mab

    return [Maa, Mab, Man, Mba, Mbb, Mbn]

def _counts_to_coefs(cnt):
    x1, x2, x4, x5 = cnt
    x3 = x1 + x2 + 1
    x6 = x4 + x5 + 1
    coefs = [x1, x2, -x3, x4, x5, -x6]
    return coefs


class GrowthFit(object):
    def __init__(self, Paa, Pbb, counts, na, prior=5):
        self.Paa = Paa
        self.Pbb = Pbb
        self.counts = counts
        self.na = na
        self.prior = prior #alpha, beta param for a beta distribution of c

    def update_counts(self, counts):
        self.counts = counts

    def loglik(self, theta):
        sa, sb, c = theta
        llik = -self.prior*(np.log(c) + np.log(1-c)) # C prior
        for paa, pbb, cnt, w in zip(self.Paa, self.Pbb, self.counts, self.coefs):
            M = _lik_catprobs(paa, pbb, self.na, sa, sb, c)
            coefs = _counts_to_coefs(cnt)
            llik -= w*sum([x * np.log(m) for x, m in zip(coefs, M)])
        return llik

    def grad_loglik(self, theta):
        sa, sb, c = theta
        grd = np.zeros(3)
        grd[2] = -self.prior*(1/theta[2] - 1/(1-theta[2]))
        for paa, pbb, cnt, w in zip(self.Paa, self.Pbb, self.counts, self.coefs):
            grd -= w*grow_gradloglikbase_jointmulti(sa, sb, c, paa, pbb, cnt, self.na)
        return grd

    def hess_loglik(self, theta):
        sa, sb, c = theta
        H = np.zeros((3,3))
        H[2, 2] = self.prior*(1/theta[2]**2 + 1/(1-theta[2])**2)
        for paa, pbb, cnt, w in zip(self.Paa, self.Pbb, self.counts, self.coefs):
            H -= w*grow_hessloglik_base(sa, sb, c, paa, pbb, cnt, self.na)
        return H

    def solve(self, x0=[.5, .5, .5], method='trust-constr'):
        bounds = optimize.Bounds([.05, 0.05, 0.05], [.95, .95, .95])
        # Check if this makes sense // to weight the likelihood to distinguish c from na
        #coefs = [1 + np.abs(.5*(1+pa-pb)-self.na)**2 for pa, pb in zip(self.Paa, self.Pbb)]
        #coefs = len(coefs)*np.array(coefs)/sum(coefs)
        #coefs = [1 if np.abs(.5*(1+pa-pb)-self.na) > np.random.rand() else 0 for pa, pb in zip(self.Paa, self.Pbb)]
        #print(sum(coefs)/len(coefs))

        coefs = np.ones(len(self.Paa))
        #coefs[::5] = 1
        self.coefs = coefs

        opt = optimize.minimize(self.loglik, x0, method=method, jac=self.grad_loglik, hess=self.hess_loglik, bounds=bounds)
        return opt

    def solve_randx0(self, n_x0=10, method='trust-constr'):
        opt = None
        fun = np.inf
        for _ in range(n_x0):
            x0 = np.random.uniform(.1, .9, 3)
            sol = self.solve(x0, method)
            if sol.fun < fun:
                fun = sol.fun
                opt = sol
        return opt

class GrowthEM(GrowthFit):
    def __init__(self, Paa, Pbb, obs_counts, na, prior=5):
        super().__init__(Paa=Paa, Pbb=Pbb, counts=None, na=na, prior=prior)
        self.obs_counts = obs_counts

    def em(self, tol=.01, theta_0=[.5, .5, .5], n_x0=2, max_iter=40):
        theta_0 = np.array(theta_0)
        i = 0
        count_dist = np.inf
        while (count_dist > tol) and (i < max_iter):
            count_dist = self.expected_counts(theta_0)
            opt = self.solve_randx0(n_x0=n_x0)
            theta_0 = opt.x #.5*opt.x + .5*theta_0
            #count_dist = self.expected_counts(sol)
            print('     sol: {} ; dist: {}'.format(theta_0, count_dist))
            i += 1
        if i >= max_iter:
            opt.success = False
        return opt

    def em_twostep(self, tol=.05, theta_0=[.5, .5, .5], max_iter=20):
        theta_0 = np.array(theta_0)
        i = 0
        count_dist = np.inf
        func = -np.inf
        while (count_dist > tol) and (i < max_iter):
            count_dist = self.expected_counts(theta_0)
            x0 = np.random.uniform(.1, .9, 3)
            opt = self.solve(x0)
            theta_0 = opt.x
            print('     sol: {} ; dist: {}'.format(theta_0, count_dist))
            i += 1
            theta_0[2] = self._c_cand(count_dist, theta_0[2])
        cvals = np.linspace(.1, .9, 9)
        ss = theta_0[:2]
        funcs = []
        for cval in cvals:
            self.expected_counts([ss[0], ss[1], cval])
            opt = self.solve(np.random.uniform(.1, .9, 3))
            self.expected_counts(opt.x)
            opt = self.solve()
            funcs.append([opt.fun, opt.x])
            print('      ---- {}: {}'.format(opt.x, opt.fun))
        return funcs

    def _c_cand(self, count_dist, ccurr):
        rs = np.sqrt(expit(count_dist) -.5)
        cand = np.inf
        while (cand > 1) or (cand < 0):
            cand = ccurr + np.random.uniform(-rs, rs)
        return cand

    def expected_counts(self, theta):
        sa, sb, c = theta
        if self.counts is not None:
            new_counts = np.copy(self.counts)
        else:
            new_counts = np.zeros((len(self.Paa), 6))
            new_counts[:, :2] = self.obs_counts[:, :2]
            new_counts[:, 3:5] = self.obs_counts[:, 2:]
            self.counts = np.copy(new_counts)
        dist = 0
        for i, (paa, pbb, cnt) in enumerate(zip(self.Paa, self.Pbb, self.obs_counts)):
            M =  _lik_catprobs(Paa=paa, Pbb=pbb, na=self.na, sa=sa, sb=sb, c=c)
            if any(cnt[:2]) > 0:
                nval = sum(cnt[:2]) * M[2] / (1 - M[2])
                new_counts[i, 2] = nval
                dist += np.abs(nval - self.counts[i][2])
            elif any(cnt[2:]) > 0:
                nval = sum(cnt[2:]) * M[5] / (1 - M[5])
                new_counts[i, 5] = nval
                dist += np.abs(nval - self.counts[i][5])
        self.update_counts(new_counts)
        dist /= (i+1)
        return dist



def grow_gradloglikbase_jointmulti(sa, sb, c, Paa, Pbb, cnt, na):
    """
    Each vector contains the log derivaties of Maa, Mab, Man, Mba, Mbb, Mbn
    """
    Dsa = [1/sa, 1/(sa-1), (2 - 4*na + 2*c*(-1 + 2*na - Paa + Pbb))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa)), 0, 0, 0]
    Dsb = [0, 0, 0, 1/(sb-1), 1/sb, (2 - 4*na + 2*c*(-1 + 2*na - Paa + Pbb))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(na + sb - 2*na*sb))]
    Dc = [0, 0, 0, 0, 0, 0]
    Dc[0] = (1 - 2*na + Paa - Pbb)/(2*na + c*(1 - 2*na + Paa - Pbb))
    Dc[1] = (-1 + 2*na - Paa + Pbb)/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))
    Dc[2] = ((-1 + 2*na - Paa + Pbb)*(-1 + 2*sa))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))
    Dc[3] =(1 - 2*na + Paa - Pbb)/(2*na + c*(1 - 2*na + Paa - Pbb))
    Dc[4] =(-1 + 2*na - Paa + Pbb)/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))
    Dc[5] =((-1 + 2*na - Paa + Pbb)*(-1 + 2*sb))/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(-1 + na + sb - 2*na*sb))

    x1, x2, x4, x5 = cnt
    x3 = x1 + x2 + 1
    x6 = x4 + x5 + 1
    cnts = [x1, x2, -x3, x4, x5, -x6]

    Lsa = sum([x*p for x, p in zip(cnts, Dsa)]) #real likelihood
    Lsb = sum([x*p for x, p in zip(cnts, Dsb)])
    Lc = sum([x*p for x, p in zip(cnts, Dc)])
    L = np.array([Lsa, Lsb, Lc])
    return L


def grow_hessloglik_base(sa, sb, c, Paa, Pbb, cnt, na):
    """
    Hessian Matrix for the growth model
    """
    h = np.zeros((3,3))
    x1, x2, x4, x5 = cnt
    x3 = (x1 + x2 + 1)
    x6 = (x4 + x5 + 1)

    #D[dmaasa*x1 + dmabsa*x2 - dmansa*x3, sa] // Simplify // FortranForm
    h[0,0] =-(x1/sa**2) - x2/(-1 + sa)**2 + (4*(1 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))**2
    h[0,1] = 0
    #D[-x3*dmansa, c] // Simplify // FortranForm
    h[0,2] =(2*(-1 + 2*na - Paa + Pbb)*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))**2

    h[1,0] = 0
    #D[dmbbsb*x5 + dmbasb*x4 - dmbnsb*x6, sb] // Simplify // FortranForm
    h[1,1] = -(x4/sb**2) - x5/(-1 + sb)**2 + (4*(1 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2*x6)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(na + sb - 2*na*sb))**2
    #D[-x6*dmbnsb, c] // Simplify // FortranForm
    h[1,2] = (-2*(-1 + 2*na - Paa + Pbb)*x6)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sb) + 2*(na + sb - 2*na*sb))**2

    #D[-x3*dmanc, sa] // Simplify // FortranForm
    h[2,0] =(2*(-1 + 2*na - Paa + Pbb)*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) +2*(-1 + na + sa - 2*na*sa))**2
    #D[-x6*dmbnc, sb] // Simplify // FortranForm
    h[2,1] = (-2*(-1 + 2*na - Paa + Pbb)*x6)/(-2 + (2 - 2*na + c*(-1 + 2*na - Paa + Pbb))*sb)**2
    #D[dmaac*x1 + dmabc*x2 - dmanc*x3 + dmbbc*x5 + dmbac*x4 - dmbnc*x6,c] // Simplify // FortranForm
    h[2,2] =(1 - 2*na + Paa - Pbb)**2*(-(x1/(2*na + c*(1 - 2*na + Paa - Pbb))**2) -x2/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2 + ((1 - 2*sa)**2*x3)/(c*(-1 + 2*na - Paa + Pbb)*(-1 + 2*sa) + 2*(-1 + na + sa - 2*na*sa))**2 - x4/(2 - 2*na + c*(-1 + 2*na - Paa + Pbb))**2 -x5/(2*na + c*(1 - 2*na + Paa - Pbb))**2 - (sb**2*x6)/(-2 + (2 - 2*na + c*(-1 + 2*na - Paa + Pbb))*sb)**2)
    return h

