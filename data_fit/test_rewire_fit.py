import rewire_degree_fit as rdf
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import seaborn as sns
import pandas as pd

def test_rewire_fit(N=1000, n_samp=5, clen=10, n_iter=3000, deg_fit=True):
    """
    Plot a function of goodness of fit for a grid of na-sa values, fixed sb  several values of c
    """
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    sb = .75
    sas = [1/3, 1/2, 3/4, 5/6]
    nas = [1/8, 1/4, 1/3, 1/2]
    cvals = np.repeat(np.linspace(.05, .95, clen), n_samp)
    color = []
    for i, sa in enumerate(sas):
        for j, na in enumerate(nas):
            fits = np.zeros((len(cvals), 3))
            fits_false = np.zeros((len(cvals), 3))
            print(i, j)
            for k, cval in enumerate(cvals):

                figname = 'plots/test_rewire_fit_degfull_N{}.pdf'.format(N)
                x, n = rdf.get_dataset(sa=sa, sb=sb, na=na, c=cval, N=N, n_iter=n_iter)
                RF = rdf.RewireFit(x, n, na=na)
                RF.solve()
                if RF.opt.success:
                    fits[k, :] = RF.opt.x
                    #vals = gdf.c_loglik(sa, sb, x, n)
                    #c = np.argmax(vals)/50
                    #fits[k, :] = [sa, sb, c]
                cest = fits[k, 2]
                print(cval, cest)

            axs[i, j].plot([0, 1], [0, 1], alpha=.5)
            axs[i, j].scatter([sa]*len(fits), fits[:, 0], c='r')
            axs[i, j].scatter([sb]*len(fits), fits[:, 1], c='k')
            axs[i, j].scatter(cvals, fits[:, 2], c='b')
            axs[i, j].set_title('na: {} | sa: {}'.format(round(na, 2), round(sa, 2)))
            fig.savefig(figname.format(N))
    fig.tight_layout()
    fig.savefig(figname.format(N))



def test_growth_fit_heatmap(N=1000, n_samp=5, em=False):
    """
    Plot a function of goodness of fit for a grid of na-cval values, and where each plot contains an error heatmap for values of sa-sb
    """
    fig, axs = plt.subplots(5, 3, figsize=(6, 10))
    sas = np.round([1/6, 1/3, 1/2, 2/3, 5/6], 2)
    cvals = np.round([1/6, 1/3, 1/2, 2/3, 5/6], 2)
    nas = np.round([1/6, 1/3, 1/2], 2)
    for i, cval in enumerate(cvals):
        for j, na in enumerate(nas):
            fits = np.zeros((len(sas), len(sas)))
            print(i, j)
            for k, sa in enumerate(sas):
                for l, sb in enumerate(sas):
                    for _ in range(n_samp):
                        Paa, Pbb, counts = gf.get_dataset(sa=sa, sb=sb, na=na, c=cval, N=N)
                        optimizer = gf.GrowthFit(Paa=Paa, Pbb=Pbb, counts=counts, na=na)
                        opt = optimizer.solve_randx0(3)
                        print(opt.x)
                        fits[k, l] += np.abs(opt.x[2] - cval)
            data = pd.DataFrame(fits/n_samp, index=sas, columns=sas)
            sns.heatmap(data, vmin=0, vmax=1, center=0, cbar=False, annot=True, ax=axs[i, j])
            axs[i, j].set_ylabel('sa')
            axs[i, j].set_xlabel('sb')
            axs[i, j].set_title('na: {} | c: {}'.format(round(na, 2), round(cval, 2)))
            fig.savefig('test_growth_fit_heatmap_N{}.pdf'.format(N))
    fig.tight_layout()
    fig.savefig('test_growth_fit_heatmap_N{}.pdf'.format(N))


def test_likelihood_c_param(N=1000, n_samp=1000, em=False):
    """
    Test wheter similar .5(1+paa-pbb) to na is the reason why c is hard to estimate
    """
    # TODO: run test varying m (varm) and with jumping likelihood (jumpllik) for em alg
    fig, ax = plt.subplots()
    params = np.random.uniform(.1, .9, (n_samp, 4))
    errors = []
    dists = []
    for i, param in enumerate(params):
        if i % 1 == 0:
            print('{}       params={}'.format(i/n_samp, np.round(param[:3], 4)))
        na = param[3]
        if (na > .95) or (na < 0.5):
            na = np.random.uniform(0.05, 0.95)
        if em:
            figname = 'likelihood_test_em_varmk.pdf'
            Paa, Pbb, obs_counts, counts = gf.get_dataset_varm(sa=param[0], sb=param[1], na=na, c=param[2], N=N, em=True)
            optimizer = gf.GrowthEM(Paa=Paa, Pbb=Pbb, obs_counts=obs_counts, na=na)
            theta_0 = np.random.uniform(.1, .9, 3)
            opt = optimizer.em(theta_0=theta_0)
        else:
            figname = 'likelihood_test_varm.pdf' #jumping: llik randomly selects
            Paa, Pbb, counts = gf.get_dataset_varm(sa=param[0], sb=param[1], na=na, c=param[2], N=N, em=False)
            optimizer = gf.GrowthFit(Paa=Paa, Pbb=Pbb, counts=counts, na=na)
            opt = optimizer.solve()

        n_dist = [np.abs(.5*(1+pa-pb)-na) for pa, pb in zip(Paa, Pbb)]
        dists.append(np.mean(n_dist))
        errors.append(np.abs(opt.x[2]-param[2]))
        print('dist: {}  --- error: {}'.format(dists[-1], errors[-1]))
    ax.plot(dists, errors, '.')
    ax.set_xlabel(r'$D((1/2)*(1+P_{aa}-P_{bb}), n_a)$')
    ax.set_ylabel(r'D(c, c^*)')
    fig.savefig(figname)

