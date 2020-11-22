import numpy as np
import asikainen_model as apm
#import generate_homophilic_graph_nonba as nonba
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import pickle
import seaborn as sns

def obs_from_t(na, t):
    oa = (1-na)*t[0] / ((1-na)*t[0] + na*(1-t[0]))
    ob = na*t[1] / ((1-na)*(1-t[1]) + na*t[1])
    return oa, ob

def run_equalsize(rewire_type='tc_two', n_avg=10, remove_neighbor=False, n_iter=600000):
    fig, ax = plt.subplots()

    if remove_neighbor:
        sname = 'rewiring/'
    else:
        sname = 'rewiring_remove_edge/'

    savename = sname + 'equalsize_{}.pdf'.format(rewire_type)
    bias = np.arange(.5, 1.05, .05)
    cvals = [0, .3, .5, .7, 1]
    n_iter = 500000 #0
    N = 1000
    fm = .5
    p0 = [[.0035, .0035], [.0035, .0035]]
    for cval in cvals:
        taas, rho_as, rho_bs = [], [], []
        print("c={}".format(cval))
        for homs in bias:
            print("     s={}".format(homs))
            hms = [homs, homs]
            taa, rho_a, rho_b = 0, 0, 0
            for _ in range(n_avg):
                _, t, _, corr, _ = apm.run_rewiring(N, fm, cval, hms, p0, n_iter=n_iter, track_steps=n_iter, rewire_type=rewire_type, remove_neighbor=remove_neighbor)
                taa += t[0]
                rho_a += corr[0]
                rho_b += corr[1]
            taa /= n_avg; rho_a /= n_avg; rho_b /= n_avg
            taas.append(taa); rho_as.append(rho_a); rho_bs.append(rho_b)
            #oas.append(corr[0])
            #oas.append(oa[0]) #for perc. bias.

        ax.plot(bias, taas, label='c={}'.format(cval)) #ax[0]
        #ax[1].plot(bias, rho_as, label='c={}'.format(cval))
        #ax[2].plot(bias, rho_bs, label='c={}'.format(cval))
        fig.savefig(savename)
    ax.set_xlabel(r'$s$') #ax[0]
    ax.set_ylabel(r'$o_a=T_{aa}$') #ax[0]
    ax.legend(loc=0) #ax[0]
    ax.set_aspect('equal', 'box')
    #ax[1].set_xlabel(r'$s$')
    #ax[1].set_ylabel(r'$\rho_a$')
    #ax[1].legend(loc=0)
    #ax[2].set_xlabel(r'$s$')
    #ax[2].set_ylabel(r'$\rho_b$')
    #ax[2].legend(loc=0)
    fig.tight_layout()
    fig.savefig(savename)

def run_diff_homophs(grow_type='tc_two', n_avg=10, fm=.1, remove_neighbor=False, N=1000):
    """
    For a fixed c=0.95, compare different sa, sb values.
    Two plots: first has taa, tbb and (taa-taa0)/taa0, and (tbb - tbb0)/tbb0
    where taa0 implies a comparison with the baseline (ba_one)
    Second: same with rho_a, and rho_b
    ?DO rho_a0 and rho_b0?
    """
    sname = 'growing/'

    if fm <= .11:
        savename = sname + 'diff_homophs_{}.p'.format(grow_type)
    elif fm >= .49:
        savename = sname + 'diff_homophs_eqsize_{}.p'.format(grow_type)
    else:
        savename = sname + 'diff_homophs_fm{}_{}.p'.format(int(100*fm), grow_type)

    #fig, ax = plt.subplots(2, 2)
    #fig_a, ax_a = plt.subplots(2, 2)
    bias = np.arange(.5, 1.05, .05)
    names = ['taa', 'tbb', 'rho_a', 'rho_b', 'taa0', 'tbb0', 'rho_a0', 'rho_b0', 'conv_d']
    cval = .95
    N = 1000
    n_iter = N
    p0 = [[.0035, .0035], [.0035, .0035]]
    results = {n: np.zeros((len(bias), len(bias))) for n in names}
    results['cval'] = cval
    results['bias'] = bias
    results['n_iter'] = n_iter
    results['p0'] = p0
    results['N'] = N
    if grow_type == 'ba_one':
        other_grow = 'ba_two'
    elif grow_type == 'ba_two':
        other_grow = 'ba_one'
    for i, sa in enumerate(bias):
        taas, tbbs, rho_as, rho_bs = [], [], [], []
        taas0, tbbs0, rho_as0, rho_bs0 = [], [], [], []
        print("sa={}".format(sa))
        for j, sb in enumerate(bias):
            print("       sb={}".format(sb))
            hms = [sa, sb]
            taa, tbb, rho_a, rho_b, = 0, 0, 0, 0
            taa0, tbb0, rho_a0, rho_b0, = 0, 0, 0, 0
            conv_d = 0

            for _ in range(n_avg):
                _, t, _, corr, conv = apm.run_growing(N, fm, cval, hms, p0, n_iter=n_iter, track_steps=n_iter, rewire_type=grow_type, remove_neighbor=remove_neighbor)
                _, t0, _, corr0, _ = apm.run_growing(N, fm, 0, hms, p0, n_iter=n_iter, track_steps=n_iter, rewire_type=other_grow, remove_neighbor=remove_neighbor)
                taa += t[0]; tbb += t[1]; rho_a += corr[0]; rho_b += corr[1]
                taa0 += t0[0]; tbb0 += t0[1]; rho_a0 += corr0[0]; rho_b0 += corr0[1]
                conv_d += conv
            taa /= n_avg; tbb /= n_avg; rho_a /= n_avg; rho_b /= n_avg
            taa0 /= n_avg; tbb0 /= n_avg; rho_a0 /= n_avg; rho_b0 /= n_avg
            conv_d /= n_avg
            _save_results(taa, tbb, rho_a, rho_b, taa0, tbb0, rho_a0, rho_b0, results, i, j, savename, grow_type, conv_d)
    plot_rewiring_heatmap(fig, ax, results, bias)


def plot_rewiring_heatmap(results, rewire_type, savename):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    bias = np.round(results['bias'], 2)
    sns.heatmap(results['taa'], ax=ax[0, 0], xticklabels=bias, yticklabels=bias)
    sns.heatmap(results['tbb'], ax=ax[0, 1], xticklabels=bias, yticklabels=bias)
    amp_taa = (results['taa'] - results['taa0']) #/results['taa0']
    amp_tbb = (results['tbb'] - results['tbb0']) #/results['tbb0']

    amp_taa[np.abs(amp_taa) == np.inf] = np.nan
    amp_tbb[np.abs(amp_tbb) == np.inf] = np.nan

    sns.heatmap(amp_taa, ax=ax[1, 0], xticklabels=bias, yticklabels=bias, center=0)
    sns.heatmap(amp_tbb, ax=ax[1, 1], xticklabels=bias, yticklabels=bias, center=0)
    ax[0, 0].invert_yaxis()

    ax[0, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_xlabel(r'$s_b$')
    ax[1, 1].set_xlabel(r'$s_b$')

    ax[0, 0].set_title(r'$T_{aa}$')
    ax[0, 1].set_title(r'$T_{bb}$')
    ax[1, 0].set_title(r'$Amp(T_{aa})$')
    ax[1, 1].set_title(r'$Amp(T_{bb})$')

    suptitles = {'pa_one': 'PA 1', 'pa_two': 'PA 2', 'tc_one': 'TC 1', 'tc_two': 'TC 2', 'tc_four': 'TC 4', 'ba_one': 'BA 1', 'ba_two': 'BA 2'}
    fig.suptitle(suptitles[rewire_type], fontsize=14)
    fig.tight_layout()
    savename = savename.replace('homophs_', 'homophs_t_') + 'df'
    fig.savefig(savename)
    plt.close(fig)

def plot_rewiring_heatmap_cp(results, rewire_type, savename):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    bias = np.round(results['bias'], 2)
    sns.heatmap(results['rho_a'], ax=ax[0, 0], xticklabels=bias, yticklabels=bias, center=1)
    sns.heatmap(results['rho_b'], ax=ax[0, 1], xticklabels=bias, yticklabels=bias, center=1)
    amp_taa = (results['rho_a'] - results['rho_a0']) #/results['rho_a0']
    amp_tbb = (results['rho_b'] - results['rho_b0']) #/results['rho_b0']

    amp_taa[np.abs(amp_taa) == np.inf] = np.nan
    amp_tbb[np.abs(amp_tbb) == np.inf] = np.nan

    sns.heatmap(amp_taa, ax=ax[1, 0], xticklabels=bias, yticklabels=bias, center=0)
    sns.heatmap(amp_tbb, ax=ax[1, 1], xticklabels=bias, yticklabels=bias, center=0)
    ax[0, 0].invert_yaxis()

    ax[0, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_xlabel(r'$s_b$')
    ax[1, 1].set_xlabel(r'$s_b$')

    ax[0, 0].set_title(r'$\rho_a$')
    ax[0, 1].set_title(r'$\rho_b$')
    ax[1, 0].set_title(r'$Amp(\rho_a)$')
    ax[1, 1].set_title(r'$Amp(\rho_b)$')
    suptitles = {'pa_one': 'PA 1', 'pa_two': 'PA 2', 'tc_one': 'TC 1', 'tc_two': 'TC 2', 'tc_four': 'TC 4', 'ba_one': 'BA 1', 'ba_two': 'BA 2'}
    fig.suptitle(suptitles[rewire_type], fontsize=14)
    fig.tight_layout()
    savename = savename.replace('homophs_', 'homophs_rho_') + 'df'
    fig.savefig(savename)
    plt.close(fig)


def plot_convergence_heatmap(results, rewire_type, savename):
    fig, ax = plt.subplots()
    bias = np.round(results['bias'], 2)
    sns.heatmap(results['conv_d'], ax=ax, xticklabels=bias, yticklabels=bias)

    ax.invert_yaxis()

    ax.set_ylabel(r'$s_a$')
    ax.set_xlabel(r'$s_b$')

    suptitles = {'pa_one': 'PA 1', 'pa_two': 'PA 2', 'tc_one': 'TC 1', 'tc_two': 'TC 2', 'tc_four': 'TC 4', 'ba_one': 'BA 1', 'ba_two': 'BA 2'}
    fig.suptitle(suptitles[rewire_type] + ' (converg)', fontsize=14)
    fig.tight_layout()
    savename = savename.replace('homophs_', 'homophs_convd_') + 'df'
    fig.savefig(savename)
    plt.close(fig)

def _save_results(taa, tbb, rho_a, rho_b, taa0, tbb0, rho_a0, rho_b0, results, i, j, savename, rewire_type, conv_d):
    results['taa'][i, j] = taa
    results['tbb'][i, j] = tbb
    results['rho_a'][i, j] = rho_b
    results['rho_b'][i, j] = rho_a
    results['taa0'][i, j] = taa0
    results['tbb0'][i, j] = tbb0
    results['rho_a0'][i, j] = rho_a0
    results['rho_b0'][i, j] = rho_b0
    results['conv_d'][i, j] = conv_d

    pickle.dump(results, open(savename, 'wb'))

    plot_rewiring_heatmap(results, rewire_type, savename)
    plot_rewiring_heatmap_cp(results, rewire_type, savename)
    plot_convergence_heatmap(results, rewire_type, savename)


def read_results_plot_hmaps(rewire_type, fm, update_a=None):
    """
    Read files generated by run_diff_homophs and make the plots.
    """
    if fm <= .11:
        savename = 'rewiring/diff_homophs_{}.p'.format(rewire_type)
    elif fm >= .49:
        savename = 'rewiring/diff_homophs_eqsize_{}.p'.format(rewire_type)
    else:
        savename = 'rewiring/diff_homophs_fm{}_{}.p'.format(int(100*fm), rewire_type)

    results = pickle.load(open(savename, 'rb'))
    if update_a is not None:
        new_rho = update_cp(results['taa'], results['tbb'], update_a)
        results['rho_a'] = new_rho[0]
        results['rho_b'] = new_rho[1]

        new_rho = update_cp(results['taa0'], results['tbb0'], update_a)
        results['rho_a0'] = new_rho[0]
        results['rho_b0'] = new_rho[1]

    plot_rewiring_heatmap(results, rewire_type, savename)
    plot_rewiring_heatmap_cp(results, rewire_type, savename)


def plot_all_hmaps():
    rewires = ['pa_one', 'pa_two', 'tc_one', 'tc_two']
    fms = [.1, .5]
    for rt in rewires:
        for fm in fms:
            try:
                read_results_plot_hmaps(rt, fm)
            except:
                print('Failed: {} - {}'.format(rt, fm))


def update_cp(taa, tbb, update_a):
    new_rho_a = np.zeros(taa.shape)
    new_rho_b = np.zeros(taa.shape)
    for i in range(taa.shape[0]):
        for j in range(taa.shape[1]):
            rho_a, rho_b = apm.measure_core_periph(taa[i, j], tbb[i, j], update_a)
            new_rho_a[i, j] = rho_a
            new_rho_b[i, j] = rho_b
    return new_rho_a, new_rho_b


def compare_tmats(fn=.15, savepath='rewiring/', nruns=10):
    fig, axs = plt.subplots(1, 2)
    bias = np.arange(.5, 1.05, .05)
    cvals = [0, .3, .5, .7, 1]
    n_iter = 400000
    N = 1000
    fm = .15
    p0 = [[.05, .05], [.05, .05]]
    for cval in cvals:
        taas = []
        tbbs = []
        print("c={}".format(cval))
        for homs in bias:
            hms = [homs, homs]
            tha, thb = 0, 0
            for run in range(nruns):
                _, t, _ = apm.run_rewiring(N, fm, cval, hms, p0, n_iter=n_iter, track_steps=n_iter)
                tha += t[0]
                thb += t[1]
            taas.append(tha/nruns)
            tbbs.append(thb/nruns)
        axs[0].plot(bias, taas, label='c={}'.format(cval))
        axs[1].plot(bias, tbbs, label='c={}'.format(cval))
    axs[0].set_xlabel(r'$s$')
    axs[0].set_ylabel(r'$T_{aa}$')
    axs[0].legend(loc=0)
    axs[1].set_xlabel(r'$s$')
    axs[1].set_ylabel(r'$T_{bb}$')
    axs[1].legend(loc=0)
    fig.tight_layout()
    fig.savefig(savename)
    plt.close(fig)

def theo_rewiring(fm, sim):
    plink = lambda fm, sim: sim * fm**2 + 2*fm*(1-fm)*(1-sim) + sim*(1-fm)**2
    paa = fm**2 * sim / plink(fm, sim)
    pab = 2 * fm * (1 - fm) * (1-sim) / plink(fm, sim)
    pbb = (1-fm) ** 2 * sim / plink(fm, sim)
    return paa, pab, pbb


def test_theo(nruns=20):
    """
    Test whether the theoretical distibution suffices for our random model
    """
    fms = np.arange(.05, .55, .05)
    sims = np.arange(.5, 1.05, .05)

    pvals_taa = np.zeros((len(fms), len(sims)))
    pvals_tbb = np.zeros((len(fms), len(sims)))

    N=700
    p0 = [[.07, .07], [.07, .07]]
    n_iter = 325000
    cval = 0
    for i, fm in enumerate(fms):
        print('fm={}'.format(fm))
        for j, sim in enumerate(sims):
            taa, tbb = [], []
            hms = [sim, sim]
            for _ in range(nruns):

                _, t, _ = apm.run_rewiring(N, fm, cval, hms, p0, n_iter=n_iter, track_steps=n_iter)
                taa.append(t[0])
                tbb.append(t[1])
            ps = theo_rewiring(fm, sim)
            taa_theo = 2 * ps[0] / (2 * ps[0] + ps[1])
            tbb_theo = 2 * ps[2] / (2 * ps[2] + ps[1])
            pvals_taa[i, j] = ttest_1samp(taa, taa_theo)[1]
            pvals_tbb[i, j] = ttest_1samp(tbb, tbb_theo)[1]
            pickle.dump(pvals_taa, open('rewiring/pvals_theo_taa.p', 'wb'))
            pickle.dump(pvals_tbb, open('rewiring/pvals_theo_tbb.p', 'wb'))
    return pvals_taa, pvals_tbb


def plot_theo_random(savename='rewiring/samp_comparison_nocorr.pdf'):
    """
    Plot theoretical values for perception bias using only homophily as presented by
    karimi and
    """
    fig, axs = plt.subplots(1, 2)
    axs[0].axhline(1, c='k', alpha=.2)
    axs[1].axhline(1, c='k', alpha=.2)
    homs = np.arange(.05, 1.05, .05)
    fms = [.1, .2, .3, .4, .5]
    cols = {.1: 'b', .2: 'm', .3: 'g', .4: 'r', .5: 'k' }
    for fm in fms:
        taa, taaba, tba, tbaba = [], [], [], []
        for sim in homs:
            ps = theo_rewiring(fm, sim)
            t_aa = 2 * ps[0] / (2 * ps[0] + ps[1])
            t_bb = 2 * ps[2] / (2 * ps[2] + ps[1])
            taa.append(t_aa) #/fm)
            tba.append((1-t_bb)) #/fm)

            # Theoretical model for karimi
            ps = nonba.theo_sel_prob(fm, sim)
            p_aa = fm * ps[0]
            p_ab = fm * ps[1] + (1-fm) * ps[2]
            p_bb = (1-fm) * ps[3]
            t_aa = 2 * p_aa / (2 * p_aa + p_ab)
            t_bb = 2 * p_bb / (2 * p_bb + p_ab)

            taaba.append(t_aa) #/fm)
            tbaba.append((1-t_bb)) #/fm)

        axs[0].plot(homs, taa, label='fm={}'.format(fm), c=cols[fm])
        axs[1].plot(homs, tba, label='fm={}'.format(fm), c=cols[fm])

        axs[0].plot(homs, taaba, ls=':', c=cols[fm])
        axs[1].plot(homs, tbaba, ls=':', c=cols[fm])

    axs[0].set_xlabel(r'$s$')
    axs[0].set_ylabel(r'$T_{aa}$')
    axs[0].legend(loc=0)
    axs[1].set_xlabel(r'$s$')
    axs[1].set_ylabel(r'$T_{ba}$')
    axs[1].legend(loc=0)
    fig.tight_layout()
    fig.savefig(savename)
    plt.close(fig)

def test_amp_diff_hom(fm=.15, savepath='rewiring/', nruns=10):
    """
    Fixing majority homophily to 0.5, we test whether a minority also has
homophily amplification or core-periphery structure
    """
    fig, axs = plt.subplots(1, 2)
    bias = np.arange(.5, 1.05, .07)
    cvals = [0, .3, .5, .7, 1]
    n_iter = 250000
    N = 1000
    p0 = [[.05, .05], [.05, .05]]
    savename = savepath + "amp_diff_hom_fm{}.pdf".format(int(100*fm))
    for cval in cvals:
        taas = []
        tbbs = []
        print("c={}".format(cval))
        for homs in bias:
            hms = [homs, .5]
            tha, thb = 0, 0
            for run in range(nruns):
                _, t, _ = apm.run_rewiring(N, fm, cval, hms, p0, n_iter=n_iter, track_steps=n_iter)
                tha += t[0]
                thb += t[1]
            taas.append(tha/nruns)
            tbbs.append(thb/nruns)
        axs[0].plot(bias, taas, label='c={}'.format(cval))
        axs[1].plot(bias, tbbs, label='c={}'.format(cval))
        fig.savefig(savename)
    axs[0].set_xlabel(r'$s$')
    axs[0].set_ylabel(r'$T_{aa}$')
    axs[0].legend(loc=0)
    axs[1].set_xlabel(r'$s$')
    axs[1].set_ylabel(r'$T_{bb}$')
    axs[1].legend(loc=0)
    fig.tight_layout()
    fig.savefig(savename)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--grow_type', type=str, default='ba_two')
    parser.add_argument('--analysis', type=str, default='diffhomo')
    parser.add_argument('--n_avg', type=int, default=5)
    parser.add_argument('--N', type=int, default=1000)
    parser.add_argument('--remove_neighbor', type=bool, default=False)
    parser.add_argument('--fm', type=float, default=0.5)

    pargs = parser.parse_args()
    assert pargs.analysis in ['eqhomo', 'diffhomo'], "Analysis must be eqhomo or diffhomo"

    if pargs.analysis == 'eqhomo':
        run_equalsize(rewire_type=pargs.grow_type, n_avg=pargs.n_avg, remove_neighbor=pargs.remove_neighbor, N=pargs.N)
    elif pargs.analysis == 'diffhomo':
        run_diff_homophs(grow_type=pargs.grow_type, n_avg=pargs.n_avg, fm=pargs.fm, remove_neighbor=pargs.remove_neighbor, N=pargs.N)


    #for rt in rewire_types:
    #    print('Rewire_tyep: {}'.format(rt))
    #    run_equalsize(rewire_type='tc_one', n_avg=10, remove_neighbor=False)
    #compare_tmats()
#    for fm in [.15, .3, .45]:
#        test_amp_diff_hom(fm)
    #run_diff_homophs(rewire_type='', n_avg=2, fm=.1)
    #run_diff_homophs(rewire_type='pa_two', n_avg=2, fm=.5)

