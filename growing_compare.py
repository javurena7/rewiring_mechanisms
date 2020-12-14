import numpy as np
import asikainen_model as apm
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import pickle
import seaborn as sns

def obs_from_t(na, t):
    oa = (1-na)*t[0] / ((1-na)*t[0] + na*(1-t[0]))
    ob = na*t[1] / ((1-na)*(1-t[1]) + na*t[1])
    return oa, ob

def run_diff_homophs(grow_type='ba_two', n_avg=10, fm=.1, remove_neighbor=False, N=1000):
    """
    Compare different sa, sb values by growing a network with grow_type.
    Three plots: first has taa, tbb; taa-taa0, tbb - tbb0; and taa-taa_bm, tbb-tbb_bm
    where taa0 implies a comparison with the baseline (BA model with no homoph) and taa_bm is an SBM with homophily
    Second: same with rho_a, and rho_b (measure of core-periphery)

    """
    sname = 'grow_compare/'

    #if fm <= .11:
    #    savename = sname + 'diff_homophs_{}.p'.format(grow_type)
    if fm >= .49:
        savename = sname + 'baseline_diff_homophs_eqsize_{}.p'.format(grow_type)
    else:
        savename = sname + 'baseline_diff_homophs_fm{}_{}.p'.format(int(100*fm), grow_type)


    bias = np.arange(.5, 1.05, .05) #np.arange(.5, 1.1, .1)
    bias = np.round(bias, 2)
    names = ['taa', 'tbb', 'rho_a', 'rho_b', 'taa_bm', 'tbb_bm', 'rho_a_bm', 'rho_b_bm', 'taa0', 'tbb0', 'rho_a0', 'rho_b0', 'conv_d'] #0 is for ba without homoph and bm for the sbm
    cval = .95
    m = 2
    n_iter = N
    p0 = [[.0035, .0035], [.0035, .0035]]
    results = {n: np.zeros((len(bias), len(bias))) for n in names}
    #results['cval'] = cval
    results['bias'] = bias
    results['m'] = m
    #results['n_iter'] = n_iter
    #results['p0'] = p0
    results['N'] = N
    Na = int(N * fm)
    other_grow = 'ba_zero'
    for i, sa in enumerate(bias):
        taas, tbbs, rho_as, rho_bs = [], [], [], [] #For growth mech
        taas0, tbbs0, rho_as0, rho_bs0 = [], [], [], [] #For PA w/o homoph
        taas_bm, tbbs_bm, rho_as_bm, rho_bs_bm = [], [], [], [] #Homoph sbm
        print("sa={}".format(sa))
        for j, sb in enumerate(bias):
            print("       sb={}".format(sb))
            hms = [sa, sb]
            taa, tbb, rho_a, rho_b, = 0, 0, 0, 0 #Growth mechanism
            taa0, tbb0, rho_a0, rho_b0, = 0, 0, 0, 0 #PA wo homoph
            taa_bm, tbb_bm, rho_a_bm, rho_b_bm, = 0, 0, 0, 0 #Homoph sbm
            conv_d = 0

            for _ in range(n_avg):
                _, t, _, corr, conv = apm.run_growing(N, fm, cval, hms, p0, n_iter=n_iter, track_steps=n_iter, rewire_type=grow_type, remove_neighbor=remove_neighbor, m=m)
                _, t0, _, corr0, _ = apm.run_growing(N, fm, 0, hms, p0, n_iter=n_iter, track_steps=n_iter, rewire_type=other_grow, remove_neighbor=remove_neighbor, m=m)
                tbm, corrbm = apm.run_hsbm(N=N, Na=Na, sa=sa, sb=sb, m=m)
                taa += t[0]; tbb += t[1]; rho_a += corr[0]; rho_b += corr[1]
                taa0 += t0[0]; tbb0 += t0[1]; rho_a0 += corr0[0]; rho_b0 += corr0[1]
                conv_d += conv
                taa_bm += tbm[0]; tbb_bm += tbm[1]; rho_a_bm += corrbm[0]; rho_b_bm += corrbm[1]
            taa /= n_avg; tbb /= n_avg; rho_a /= n_avg; rho_b /= n_avg
            taa0 /= n_avg; tbb0 /= n_avg; rho_a0 /= n_avg; rho_b0 /= n_avg
            taa_bm /= n_avg; tbb_bm /= n_avg; rho_a_bm /= n_avg; rho_b_bm /= n_avg
            conv_d /= n_avg
            _save_results(taa=taa, tbb=tbb, rho_a=rho_a, rho_b=rho_b, taa0=taa0, tbb0=tbb0, rho_a0=rho_a0, rho_b0=rho_b0, taa_bm=taa_bm, tbb_bm=tbb_bm, rho_a_bm=rho_a_bm, rho_b_bm=rho_b_bm, results=results, i=i, j=j, savename=savename, grow_type=grow_type, conv_d=conv_d)
    #plot_rewiring_heatmap(fig, ax, results, bias)


def plot_rewiring_heatmap(results, rewire_type, savename):
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    bias = np.round(results['bias'], 2)
    sns.heatmap(results['taa'], ax=ax[0, 0], xticklabels=bias, yticklabels=bias)
    sns.heatmap(results['tbb'], ax=ax[0, 1], xticklabels=bias, yticklabels=bias)
    amp_taa = (results['taa'] - results['taa0']) #/results['taa0']
    amp_tbb = (results['tbb'] - results['tbb0']) #/results['tbb0']

    amp_taa[np.abs(amp_taa) == np.inf] = np.nan
    amp_tbb[np.abs(amp_tbb) == np.inf] = np.nan

    sns.heatmap(amp_taa, ax=ax[1, 0], xticklabels=bias, yticklabels=bias, center=0)
    sns.heatmap(amp_tbb, ax=ax[1, 1], xticklabels=bias, yticklabels=bias, center=0)
    amp_taa_bm = (results['taa'] - results['taa_bm']) #/results['taa0']
    amp_tbb_bm = (results['tbb'] - results['tbb_bm']) #/results['tbb0']

    amp_taa_bm[np.abs(amp_taa_bm) == np.inf] = np.nan
    amp_tbb_bm[np.abs(amp_tbb_bm) == np.inf] = np.nan

    sns.heatmap(amp_taa_bm, ax=ax[2, 0], xticklabels=bias, yticklabels=bias, center=0)
    sns.heatmap(amp_tbb_bm, ax=ax[2, 1], xticklabels=bias, yticklabels=bias, center=0)

    ax[0, 0].invert_yaxis()

    ax[0, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_ylabel(r'$s_a$')
    ax[2, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_xlabel(r'$s_b$')
    ax[1, 1].set_xlabel(r'$s_b$')
    ax[2, 1].set_xlabel(r'$s_b$')

    ax[0, 0].set_title(r'$T_{aa}$')
    ax[0, 1].set_title(r'$T_{bb}$')
    ax[1, 0].set_title(r'$T_{aa} - T^{PA}_{aa}$')
    ax[1, 1].set_title(r'$T_{bb} - T^{PA}_{bb}$')
    ax[2, 0].set_title(r'$T_{aa} - T^{SBM}_{aa}$')
    ax[2, 1].set_title(r'$T_{bb} - T^{SBM}_{bb}$')

    suptitles = {'pa_one': 'PA 1', 'pa_two': 'PA 2', 'tc_one': 'TC 1', 'tc_two': 'TC 2', 'tc_four': 'TC 4', 'ba_one': 'BA 1', 'ba_two': 'BA 2'}
    fig.suptitle(suptitles[rewire_type], fontsize=14)
    fig.tight_layout()
    savename = savename.replace('homophs_', 'homophs_t_') + 'df'
    fig.savefig(savename)
    plt.close(fig)

def plot_rewiring_heatmap_cp(results, rewire_type, savename):
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    bias = np.round(results['bias'], 2)
    sns.heatmap(results['rho_a'], ax=ax[0, 0], xticklabels=bias, yticklabels=bias, center=1)
    sns.heatmap(results['rho_b'], ax=ax[0, 1], xticklabels=bias, yticklabels=bias, center=1)
    amp_taa = (results['rho_a'] - results['rho_a0']) #/results['rho_a0']
    amp_tbb = (results['rho_b'] - results['rho_b0']) #/results['rho_b0']

    amp_taa[np.abs(amp_taa) == np.inf] = np.nan
    amp_tbb[np.abs(amp_tbb) == np.inf] = np.nan

    sns.heatmap(amp_taa, ax=ax[1, 0], xticklabels=bias, yticklabels=bias, center=0)
    sns.heatmap(amp_tbb, ax=ax[1, 1], xticklabels=bias, yticklabels=bias, center=0)

    amp_taa_bm = (results['rho_a'] - results['rho_a_bm']) #/results['taa0']
    amp_tbb_bm = (results['rho_b'] - results['rho_b_bm']) #/results['tbb0']

    amp_taa_bm[np.abs(amp_taa_bm) == np.inf] = np.nan
    amp_tbb_bm[np.abs(amp_tbb_bm) == np.inf] = np.nan

    sns.heatmap(amp_taa_bm, ax=ax[2, 0], xticklabels=bias, yticklabels=bias, center=0)
    sns.heatmap(amp_tbb_bm, ax=ax[2, 1], xticklabels=bias, yticklabels=bias, center=0)

    ax[0, 0].invert_yaxis()

    ax[0, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_ylabel(r'$s_a$')
    ax[1, 0].set_xlabel(r'$s_b$')
    ax[1, 1].set_xlabel(r'$s_b$')

    ax[0, 0].set_title(r'$\rho_{a}$')
    ax[0, 1].set_title(r'$\rho_{b}$')
    ax[1, 0].set_title(r'$\rho_{a} - \rho^{PA}_{a}$')
    ax[1, 1].set_title(r'$\rho_{b} - \rho^{PA}_{b}$')
    ax[2, 0].set_title(r'$\rho_{a} - \rho^{SBM}_{a}$')
    ax[2, 1].set_title(r'$\rho_{b} - \rho^{SBM}_{b}$')

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

def _save_results(taa, tbb, rho_a, rho_b, taa0, tbb0, rho_a0, rho_b0, taa_bm, tbb_bm, rho_a_bm, rho_b_bm, results, i, j, savename, grow_type, conv_d):
    results['taa'][i, j] = taa
    results['tbb'][i, j] = tbb
    results['rho_a'][i, j] = rho_a
    results['rho_b'][i, j] = rho_b
    results['taa0'][i, j] = taa0
    results['tbb0'][i, j] = tbb0
    results['rho_a0'][i, j] = rho_a0
    results['rho_b0'][i, j] = rho_b0
    results['taa_bm'][i, j] = taa_bm
    results['tbb_bm'][i, j] = tbb_bm
    results['rho_a_bm'][i, j] = rho_a_bm
    results['rho_b_bm'][i, j] = rho_b_bm

    results['conv_d'][i, j] = conv_d

    pickle.dump(results, open(savename, 'wb'))

    plot_rewiring_heatmap(results, grow_type, savename)
    plot_rewiring_heatmap_cp(results, grow_type, savename)
    #plot_convergence_heatmap(results, rewire_type, savename)


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



if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--grow_type', type=str) #, default='ba_two')
    parser.add_argument('--analysis', type=str, default='diffhomo')
    parser.add_argument('--n_avg', type=int, default=2)
    parser.add_argument('--N', type=int, default=500)
    parser.add_argument('--remove_neighbor', type=bool, default=False)
    parser.add_argument('--fm', type=float, default=0.1)

    pargs = parser.parse_args()
    assert pargs.analysis in ['diffhomo'], "Analysis must be eqhomo or diffhomo"

    if pargs.analysis == 'eqhomo':
        run_equalsize(grow_type=pargs.grow_type, n_avg=pargs.n_avg, remove_neighbor=pargs.remove_neighbor, N=pargs.N)
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

