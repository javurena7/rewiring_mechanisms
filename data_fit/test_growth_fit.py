import growth_fit as gf
import matplotlib.pyplot as plt; plt.ion()
import numpy as np
import seaborn as sns
import pandas as pd

def test_growth_fit(N=1000, n_samp=5, clen=10):
    """
    Plot a function of goodness of fit for a grid of na-sa values, fixed sb  several values of c
    """
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    sb = .75
    sas = [1/6, 1/3, 1/2, 2/3, 5/6]
    nas = [1/6, 1/3, 1/2, 2/3, 5/6]
    cvals = np.repeat(np.linspace(.05, .95, clen), n_samp)
    for i, sa in enumerate(sas):
        for j, na in enumerate(nas):
            fits = np.zeros((len(cvals), 3))
            print(i, j)
            for k, cval in enumerate(cvals):
                Paa, Pbb, counts = gf.get_dataset(sa=sa, sb=sb, na=na, c=cval, N=N)
                optimizer = gf.GrowthFit(Paa=Paa, Pbb=Pbb, counts=counts, na=na)
                opt = optimizer.solve_randx0(4)
                print(opt.x)
                fits[k, :] = opt.x
            axs[i, j].plot([0, 1], [0, 1], alpha=.5)
            axs[i, j].scatter([sa]*len(fits), fits[:, 0], c='r')
            axs[i, j].scatter([sb]*len(fits), fits[:, 1], c='k')
            axs[i, j].scatter(cvals, fits[:, 2], c='b')
            axs[i, j].set_title('na: {} | sa: {}'.format(round(na, 2), round(sa, 2)))
            fig.savefig('test_growth_fit_N{}.pdf'.format(N))
    fig.tight_layout()
    fig.savefig('test_growth_fit_N{}.pdf'.format(N))



def test_growth_fit_heatmap(N=1000, n_samp=5):
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



