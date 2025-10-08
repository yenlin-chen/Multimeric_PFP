import os, wandb
import numpy as np
import matplotlib.pyplot as plt

def plot_pr(
        save_dir,
        precision,
        recall,
        thres_list,
        name=None,
        filename_suffix=None,
        wandb_run=None
    ):

    fig = plt.figure('pr', figsize=(5,5), dpi=300, constrained_layout=True)
    ax = fig.gca()

    ### F1 CONTOUR
    levels = 10
    spacing = np.linspace(0, 1, 1000)
    x, y = np.meshgrid(spacing, spacing)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = 2 / (1/x + 1/y)
    locx = np.linspace(0, 1, levels, endpoint=False)[1:]
    cs = ax.contour(x, y, f1, levels=levels, linewidths=1, colors='k',
                    alpha=0.3)
    ax.clabel(cs, inline=True, fmt='F1=%.1f',
                manual=np.tile(locx,(2,1)).T)

    # COMPUTE F1_MAX AND AUPR
    with np.errstate(divide='ignore', invalid='ignore'):
        aupr = np.trapz(np.flip(precision), x=np.flip(recall))
        f1 = 2*recall*precision / (recall+precision)

    f1_max_idx = np.nanargmax(f1)
    f1_max = f1[f1_max_idx]

    # PLOT PR CURVE
    ax.plot(recall, precision, lw=1, color='C0', label=name)

    # MARK F1 AT F1_MAX AND DEFAULT THRESHOLD (0.5)
    ax.scatter(recall[f1_max_idx], precision[f1_max_idx],
                label=f'{thres_list[f1_max_idx]:.4f} (f1$_{{max}}$)',
                marker='o', edgecolors='C1',
                facecolors='none', linewidths=0.5)
    if thres_list.size%2 == 1:
        def_thres_idx = (thres_list.size-1)//2
        ax.scatter(recall[def_thres_idx], precision[def_thres_idx],
                    label=f'{0.5:.4f}',
                    marker='x', c='C1',
                    linewidths=0.5)
    plt.legend(title='threshold', loc='lower left')

    plt.xlabel('recall')
    plt.ylabel('precision')

    # plt.title(f'AUPR: {aupr}, f1: {f1_max}')
    plt.title(f'f1$_{{max}}$: {f1_max:.6f}')

    # plt.show()
    if save_dir:
        filename = (f'pr_curve-{filename_suffix}.png'
                        if filename_suffix else 'pr_curve.png')
        plt.savefig(os.path.join(save_dir, filename))
    if wandb_run:
        wandb_run.log({f'pr {filename_suffix}': wandb.Image(fig)})

    plt.close()
