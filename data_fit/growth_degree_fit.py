import numpy as np
import sys
sys.path.append('..')
import asikainen_model as am
import sympy as sym
from scipy import optimize
from scipy.special import expit

"""
GrowthFit is in working condition for degree-based estimates
"""

def get_dataset(sa, sb, na, c, N, m=10):
    n_iter = 0
    x, n = am.run_growing_varying_m(N, na, c, [sa, sb], ['poisson', m], deg_based=True)
    return x, n

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

def _get_h(xt, sa, sb):
    if (xt.get('aa') or xt.get('ab')):
        return 'a', sa, (1-sa)
    elif (xt.get('ba') or xt.get('bb')):
        return 'b', (1-sb), sb
    else:
        return '', 0, 0


def _lik_step_t_probs(c, sa, sb, nt, xt):
    src, ha, hb = _get_h(xt, sa, sb)
    if not src:
        return 0
    Pa_tot = sum([n*k for k, n in nt['na'].items()])
    Pb_tot = sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = sum(nt['na'].values())
    Ub_tot = sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tot = 1 + sum(xt[src+'a'].values()) + sum(xt[src+'b'].values())

    llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0

    #Prob of going from src to a
    for k, xk in xt[src+'a'].items():
        na_k = nt['na'].get(k, 0)
        pia_k = c * na_k*k/P_denm #if k*P_denm > 0 else 0
        pia_k += (1-c)*(na_k/U_denm) #if U_denm > 0 else 0
        llik += xk * (np.log(pia_k) + np.log(ha)) if pia_k > 0 else xk*np.log(ha)

    #Prob of going from src to b
    for k, xk in xt[src+'b'].items():
        nb_k = nt['nb'].get(k, 0)
        pib_k = c * nb_k*k/P_denm #if k*P_denm > 0 else 0
        pib_k += (1-c)*(nb_k/U_denm) #if U_denm > 0 else 0
        llik += xk * (np.log(pib_k) + np.log(hb)) if pib_k > 0 else xk*np.log(hb)
    return llik


def _gradlik_step_t_probs(c, sa, sb, nt, xt):
    src, ha, hb = _get_h(xt, sa, sb)
    if not src:
        return 0
    L = np.zeros(3)

    Pa_tot = sum([n*k for k, n in nt['na'].items()])
    Pb_tot = sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = sum(nt['na'].values())
    Ub_tot = sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tota = 1 + sum(xt['aa'].values()) + sum(xt['ab'].values())
    x_totb = 1 + sum(xt['bb'].values()) + sum(xt['ba'].values())

    #llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0
    Ub = Ub_tot / U_denm
    Ua = Ua_tot / U_denm
    Dca = Pa_tot / P_denm - Ua
    Dcb = Pb_tot / P_denm - Ub
    DUa = Ua - Ub
    DUb = Ub - Ua
    DL_dsa_num = c*(Dca - Dcb) + DUa
    DL_dsa = DL_dsa_num / (sa * DL_dsa_num + c*Dcb + Ub)

    L[0] = 1/sa * sum(xt['aa'].values()) - (1/(1-sa))*sum(xt['ab'].values()) - x_tota*DL_dsa
    DL_dsb_num = c*(Dcb - Dca) + DUb
    DL_dsb = DL_dsb_num / (sb * DL_dsb_num + c*Dca + Ua)
    L[1] = 1/sb *sum( xt['bb'].values()) - (1/(1-sb))*sum(xt['ba'].values()) - x_totb*DL_dsb

    x_tot = 1 + sum(xt[src+'a'].values()) + sum(xt[src+'b'].values())
    if src == 'a':
        num = sa*(Dca - Dcb) + Dcb
        den = c * num + sa*DUa + Ub
    if src == 'b':
        num = sb*(Dcb - Dca) + Dca
        den = c * num + sb*DUb + Ua
    L[2] += -x_tot*num/den
    #Prob of going from src to a
    for k, xk in xt[src+'a'].items():
        na_k = nt['na'].get(k, 0)
        pa_k = na_k*k/P_denm #if k*P_denm > 0 else 0
        ua_k = na_k / U_denm
        Dca_k = pa_k - ua_k
        L[2] += xk * Dca_k / (c*Dca_k + ua_k) if c*Dca_k + ua_k > 0 else 0

    #Prob of going from src to b
    for k, xk in xt[src+'b'].items():
        nb_k = nt['nb'].get(k, 0)
        pb_k = nb_k*k/P_denm #if k*P_denm > 0 else 0
        ub_k = nb_k / U_denm
        Dcb_k = pb_k - ub_k
        L[2] += xk * Dcb_k / (c*Dcb_k + ub_k) if c*Dcb_k + ub_k > 0 else 0

    return L

def _hesslik_step_t_probs(c, sa, sb, nt, xt):
    src, ha, hb = _get_h(xt, sa, sb)
    if not src:
        return 0
    H = np.zeros((3, 3))

    Pa_tot = sum([n*k for k, n in nt['na'].items()])
    Pb_tot = sum([n*k for k, n in nt['nb'].items()])
    Ua_tot = sum(nt['na'].values())
    Ub_tot = sum(nt['nb'].values())

    P_denm = Pa_tot + Pb_tot # Denom of PA kernel
    U_denm = Ua_tot + Ub_tot # Denom of Unif kernel

    # For log(sum(pia_j + pib_j))

    P_tot = c*(Pa_tot*ha + Pb_tot*hb)/P_denm #if P_denm > 0 else 0
    P_tot += (1-c)*(Ua_tot*ha + Ub_tot*hb)/U_denm #if U_denm > 0 else 0
    x_tota = 1 + sum(xt['aa'].values()) + sum(xt['ab'].values())
    x_totb = 1 + sum(xt['bb'].values()) + sum(xt['ba'].values())

    #llik = - x_tot * np.log(P_tot) #if P_tot > 0 else 0
    Ub = Ub_tot / U_denm
    Ua = Ua_tot / U_denm
    Dca = Pa_tot / P_denm - Ua
    Dcb = Pb_tot / P_denm - Ub
    DUa = Ua - Ub
    DUb = Ub - Ua

    DL_dsa_num = (c*(Dca - Dcb) + DUa)
    DL_dsa = DL_dsa_num**2 / (sa * DL_dsa_num + c*Dcb + Ub)**2
    H[0, 0] = -1/sa**2 * sum(xt['aa'].values()) - (1/(1-sa)**2)*sum(xt['ab'].values()) + x_tota*DL_dsa
    DL_dsb_num = c*(Dcb - Dca) + DUb
    DL_dsb = DL_dsb_num**2 / (sb * DL_dsb_num + c*Dca + Ua)**2
    H[1, 1] = 1/sb * sum(xt['bb'].values()) - (1/(1-sb))*sum(xt['ba'].values()) - x_totb*DL_dsb

    H[0, 2] = ((Dca-Dcb)*Ub - Dcb*DUa) / (c*(Dcb +sa*(Dca-Dcb))+sa*DUa+Ub)**2
    H[2, 0] = H[0, 2]

    H[1, 2] = ((Dcb-Dca)*Ua - Dca*DUb) / (c*(Dca +sb*(Dcb-Dca))+sb*DUb+Ua)**2
    H[2, 1] = H[1, 2]

    x_tot = 1 + sum(xt[src+'a'].values()) + sum(xt[src+'b'].values())
    if src == 'a':
        num = sa*(Dca - Dcb) + Dcb
        den = c * num + sa*DUa + Ub
    if src == 'b':
        num = sb*(Dcb - Dca) + Dca
        den = c * num + sb*DUb + Ua
    H[2, 2] += x_tot*(num**2)/den**2
    #Prob of going from src to a
    for k, xk in xt[src+'a'].items():
        na_k = nt['na'].get(k, 0)
        pa_k = na_k*k/P_denm #if k*P_denm > 0 else 0
        ua_k = na_k / U_denm
        Dca_k = pa_k - ua_k
        H[2, 2] -= xk * (Dca_k / (c*Dca_k + ua_k))**2 if c*Dca_k + ua_k > 0 else 0

    #Prob of going from src to b
    for k, xk in xt[src+'b'].items():
        nb_k = nt['nb'].get(k, 0)
        pb_k = nb_k*k/P_denm #if k*P_denm > 0 else 0
        ub_k = nb_k / U_denm
        Dcb_k = pb_k - ub_k
        H[2, 2] -= xk * (Dcb_k / (c*Dcb_k + ub_k))**2 if c*Dcb_k + ub_k > 0 else 0

    return H

def c_loglik(sa, sb, x, n):
    vals = [loglik(cv, sa, sb, x, n) for cv in np.linspace(0, 1, 50)]
    return vals

def sa_loglik(c, sb, x, n):
    vals = [loglik(c, sa, sb, x, n) for sa in np.linspace(0, 1, 50)]
    return vals


def _counts_to_coefs(cnt):
    x1, x2, x4, x5 = cnt
    x3 = x1 + x2 + 1
    x6 = x4 + x5 + 1
    coefs = [x1, x2, -x3, x4, x5, -x6]
    return coefs


class GrowthFit(object):
    def __init__(self, x_evol, n_evol, na):
        # x_evol and n_evol are the timestamped values for x (new connections by degree and group) and n (group deg distribution)
        self.x_evol = x_evol
        self.n_evol = n_evol
        self.na = na

    def update_counts(self, counts):
        self.counts = counts

    def loglik(self, theta): #c, sa, sb, x, n):
        sa, sb, c = theta
        llik = 0
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 10:
                llik -= _lik_step_t_probs(c, sa, sb, nt, xt)
        return llik

    def grad_loglik(self, theta): #c, sa, sb, x, n):
        grad = np.zeros(3)
        sa, sb, c = theta
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 10:
                grad -= _gradlik_step_t_probs(c, sa, sb, nt, xt)
        return grad

    def hess_loglik(self, theta): #c, sa, sb, x, n):
        sa, sb, c = theta
        hess = np.zeros((3, 3))
        for (t, xt), nt in zip(self.x_evol.items(), self.n_evol.values()):
            if t > 10:
                hess -= _hesslik_step_t_probs(c, sa, sb, nt, xt)
        return hess


    def solve(self, x0=[.5, .5, .5], method='trust-constr'):
        bounds = optimize.Bounds([.05, 0.05, 0.05], [.95, .95, .95])
        opt = optimize.minimize(self.loglik, x0, method=method, jac=self.grad_loglik, hess=self.hess_loglik, bounds=bounds)
        self.opt = opt
        return opt.x

    def solve_randx0(self, n_x0=10, method='trust-constr'):
        opt = None
        fun = np.inf
        for _ in range(n_x0):
            x0 = np.random.uniform(.1, .9, 3)
            sol = self.solve(x0, method)
            if sol.fun < fun:
                fun = sol.fun
                opt = sol
        self.opt = opt
        return opt.x

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

