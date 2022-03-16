import matplotlib.pyplot as plt; plt.ion()
import sys
sys.path.append('..')
import get_fixed_points as gfp
import numpy as np

lsts = {0: '-',
        1: '-',
        2: '-',
        3: '--',
        4: '--'}

def plot_fullrange_policy_rewire(x='sa', params={'c':0.95, 'na':.1, 'sb':.5}, rho=0.1, obs_0={}, obs_n={}, title='', fig=None, ax=None, xlims=(0, 1), data=None, nsize=100):

    if ax is None:
        fig, ax = plt.subplots(figsize=(4,4))

    if not data:
        data = get_pred_vals_rewire(x, params, xlims, rho, nsize)
    alims, blims, ilims = get_core_limits(data)
    for lim in alims:
        ax.fill_betweenx([-1, 1], *lim, color='orangered', alpha=.2)
    for lim in blims:
        ax.fill_betweenx([-1, 1], *lim, color='royalblue', alpha=.2)
    for lim in ilims:
        ax.fill_betweenx([-1, 1], *lim, color='magenta', alpha=.2)

    for i, line in enumerate(data):
        if np.any(line) and i < 3:
            xs = line[:, 0]
            ax.plot(xs, line[:, 1], lsts[i], color='orangered')#, alpha=.6)
            ax.plot(xs, line[:, 2], lsts[i], color='royalblue')#, alpha=.6)
        if np.any(line) and i == 3:
            xs = line[:, 0]
            ax.plot(xs, line[:, 1], lsts[i], color='maroon')
        if np.any(line) and i == 4:
            xs = line[:, 0]
            ax.plot(xs, line[:, 1], lsts[i], color='darkblue')
    ax.set_xlim(xlims)
    if (obs_n) and (obs_0):
        plot_observed(x, obs_0, obs_n, ax)
    ylims = (-.2, 1)
    ax.set_ylim(ylims)
    return data


def plot_observed(x, obs0, obsn, ax):
    pa, pb = obs0.get('paa'), obs0.get('pbb')
    pan, pbn = obsn.get('paa'), obsn.get('pbb')

    xv = obs0.get(x)
    ax.axhline(pa, ls='--', alpha=.5, color='orangered')
    ax.axhline(pb, ls='--', alpha=.5, color='royalblue')
    ax.quiver(xv, pa, 0, pan-pa, units='y', scale=1, alpha=.5, color='orangered')
    ax.quiver(xv, pb, 0, pbn-pb, units='y', scale=1, alpha=.5, color='royalblue')
    ax.plot(xv, pa, 'o', color='orangered')
    ax.plot(xv, pan, 'X', color='orangered')
    ax.plot(xv, pb, 'o', color='royalblue')
    ax.plot(xv, pbn, 'X', color='royalblue')

def plot_policies(x, obs0, pols, ax):
    ### pols = list of format [x, [(pa0, pb0), (pa1, pb1), ..., (pan, pbn)]]
    pa, pb = obs0.get('paa'), obs0.get('pbb')
    t_size = 2500
    time_unit = 500
    for pol in pols:
        pa0, pb0 = pa, pb
        for yr in pol[1]:
            ax.quiver(pol[0], pa, 0, yr[1]-pa0, units='y', scale=1, alpha=.5, color='orangered')
            ax.quiver(pol[0], pb, 0, yr[2]-pb0, units='y', scale=1, alpha=.5, color='orangered')
            pa0, pb0 = yr[1], yr[2]




def get_pred_vals_rewire(x, params, lims, rho, nsize=100):
    xvals = np.linspace(*lims, nsize+2)[1:-1]
    preds = [[], [], [], [], []] #unique, topstable, lowstable, r_acore, r_bcore
    for xv in xvals:
        params[x] = xv
        na = params['na']
        sols = gfp.rewire_fixed_points(**params)
        classify_rewire_sols(sols, na, rho, preds, xv)

    #preds[1].insert(0, preds[0][-1])
    #preds[2].insert(0, preds[0][-1])
    preds = [np.array(pred) for pred in preds]
    return preds


def classify_rewire_sols(sols, na, rho, preds, xv):
    #TODO: add classification of rho's (by who is the core)
    #import pdb; pdb.set_trace()
    if len(sols) == 1:
        rx, grp, core_status = check_core(sols[0], na, rho)
        #unique
        preds[0].append([xv, sols[0][0], sols[0][1]])
        if grp == 'a':
            preds[3].append([xv, rx, core_status])
        elif grp == 'b':
            preds[4].append([xv, rx, core_status])

    elif len(sols) == 3:
        preds[1].append([xv, sols[0][0], sols[0][1]])
        preds[2].append([xv, sols[2][0], sols[2][1]])
        for i in [0, 2]:
            rx, grp, core_status = check_core(sols[i], na, rho)
            if grp == 'a':
                preds[3].append([xv, rx, core_status])
            elif grp == 'b':
                preds[4].append([xv, rx, core_status])

def check_core(sol, na, rho):
    ra = gfp.cp_correlation_cont(sol[0], sol[1], na, rho, 0.01)
    rb = gfp.cp_correlation_cont(sol[1], sol[0], 1-na, rho, 0.01)

    if (ra > 0) and (ra > rb):
        rhoa, rhoab, rhob = get_local_densities(rho, na, sol[0], sol[1])
        cnd = 1 if (rhoa > rhoab) and (rhoab > rhob) else 0
        return ra, 'a', cnd
    elif (rb > 0) and (rb > ra):
        rhoa, rhoab, rhob = get_local_densities(rho, na, sol[0], sol[1])
        cnd = 1 if (rhob > rhoab) and (rhoab > rhoa) else 0
        return rb, 'b', cnd
    else:
        return 0, '', 0

def get_local_densities(rho, na, paa, pbb):
    rho_aa = paa * rho / na**2
    rho_bb = pbb * rho / (1-na)**2
    rho_ab = (1-paa-pbb) * rho / (na*(1-na)) #TODO check if correct
    #print(rho_aa, rho_ab, rho_bb)
    return rho_aa, rho_ab, rho_bb


def get_core_limits(data):
    ## NOTE: this assumes that there can only be one intersection (acore and bcore are two continious ranges)
    acore = data[3]
    bcore = data[4]
    acore_lims = _core_limits(acore) if np.any(acore) else []
    bcore_lims = _core_limits(bcore) if np.any(bcore) else []
    print(acore_lims)
    print(bcore_lims)
    #TODO: add case where there no core or only one core
    if (not acore_lims) and (not bcore_lims):
        return [], [], []
    elif (acore_lims) and (not bcore_lims):
        return [acore_lims], [], []
    elif (bcore_lims) and (not acore_lims):
        return [], [bcore_lims], []
    elif acore_lims[0] < bcore_lims[0]:
        alims, blims, ilims = get_intersection(acore_lims, bcore_lims)
        return alims, blims, ilims
    elif bcore_lims[0] < acore_lims[0]:
        blims, alims, ilims = get_intersection(bcore_lims, acore_lims)
        return alims, blims, ilims
    else:
        return [], [], []


def get_intersection(acore_lims, bcore_lims):
    alims, blims, ilims = [], [], []
    if acore_lims[0] < bcore_lims[0]:
        if acore_lims[1] < bcore_lims[0]:
            alims = [acore_lims]
            blims = [bcore_lims]
        else:
            alims = [[acore_lims[0], bcore_lims[0]]]
            if acore_lims[1] < bcore_lims[1]:
                ilims = [[bcore_lims[0], acore_lims[1]]]
                blims = [[acore_lims[1], bcore_lims[1]]]
            else:
                ilims = [bcore_lims]
                alims += [[bcore_lims[1], acore_lims[1]]]

    return alims, blims, ilims

def _core_limits(acore):
    acore_x = []
    curr = acore[0, 2]
    if acore[0, 2] == 1:
        acore_x.append(acore[0, 0])
    for i in range(1, acore.shape[0]):
        if acore[i, 2] != curr:
            acore_x.append(acore[i, 0])
            curr = acore[i, 2]
    if curr == 1:
        acore_x.append(acore[-1, 0]) #acore[i, 0])
#    if acore[-1, 0] != last_x:
#        acore_x.append()
    return acore_x


