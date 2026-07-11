#!/usr/bin/env python3

from pyexpat import model
import sys,glob,os

num_cores = 1
os.environ["OMP_NUM_THREADS"] = f"{num_cores}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{num_cores}"
os.environ["MKL_NUM_THREADS"] = f"{num_cores}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{num_cores}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{num_cores}"


import numpy as np
import time
import rbpnfe


import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Rectangle
import pandas as pd

from matplotlib.colors import TwoSlopeNorm
from matplotlib import cm
import matplotlib as mpl
# Leave text as text in the SVG
mpl.rcParams['svg.fonttype'] = 'none'
# (Optional) choose a font you have installed:
mpl.rcParams['font.family'] = 'Arial'

def cm_to_inch(cm: float) -> float:
    return cm/2.54

##################################################
# general Figure Setup

axlinewidth  = 0.5
axtick_major_width  = 0.5
axtick_major_length = 2.0
axtick_minor_width  = 0.4
axtick_minor_length = 1.2

tick_pad        = 2
tick_labelsize  = 5
label_fontsize  = 6
legend_fontsize = 6

panel_label_fontsize = 8
label_fontweight= 'bold'
panel_label_fontweight= 'bold'

##################################################
# colors 

colors = ['#8BCDE5','#189BCC','#0C4D66','#B2C792','#668F25','#334712']
colors = ['#8BCDE5','#189BCC','#0C4D66','#F3C18E','#EC9943','#8D5B28']
colors = ['#0f3a58', '#16537e', '#5b86a4','', '', '']
colors = ['#0f3a58', '#16537e', '#5b86a4','#F3C18E','#EC9943','#8D5B28']
colors = ['#7fcae4','#0096c9','#004b64','#623b12','#C57624','#e2ba91']

base_markers = ['o','s','D']
base_markers = ['s','o','D']
base_size    = [18,13,13.2]

##################################################
# Plot Specs

markers = ['o','s','D','<']
base_size = [18,13,13.2]

filled_markers = True
scatter_zorder = 3
scatter_alpha  = 0.8
plot_zorder    = 1

theory_ls = '-'
theory_alpha = 1.0

marker_size     = 10
marker_linewidth = 0.7
plot_linewidth  = 1


fig_width = 8.0
fig_height = 10.6

fig = plt.figure(figsize=(cm_to_inch(fig_width), cm_to_inch(fig_height)), dpi=300,facecolor='w',edgecolor='k') 
axes = []
axes.append(plt.subplot2grid(shape=(3, 1), loc=(0, 0), colspan=1,rowspan=1))
plt.minorticks_on() 
axes.append(plt.subplot2grid(shape=(3, 1), loc=(1, 0), colspan=1,rowspan=1))
plt.minorticks_on() 
axes.append(plt.subplot2grid(shape=(3, 1), loc=(2, 0), colspan=1,rowspan=1))
plt.minorticks_on() 

ax1 = axes[0]
ax2 = axes[1]
ax3 = axes[2]

nbp = 1147
seq = ''.join(['ATCG'[np.random.randint(4)] for i in range(nbp)])

# seq = "CG"*125
# seq = "AT"*125

##########################################################################################################################################

params_model = 'md'
hard_constraint = True
ncores = 4
verbose = False
use_correction = True

shl_open_left = 0
shl_open_right = 0

factors = None

nfe = rbpnfe.NucFreeEnergy(
    params_model = params_model,
    hardconstraint=hard_constraint,
    rescale_factors=factors
    )


t1 = time.time()
fes_md = nfe.eval_landscape(
    seq,
    shl_open_left = shl_open_left,
    shl_open_right = shl_open_right,
    use_correction = use_correction,
    ncores = ncores,
    verbose = verbose
    )
t2 = time.time()
print(f'Time elapsed {t2-t1:.4f} s') 

indices = np.arange(len(fes_md))
color = colors[0]
label = 'MD'
ax1.plot(indices, fes_md[:,0], color=color, lw=plot_linewidth, ls='-', alpha=theory_alpha, zorder=plot_zorder, label=label)
ax2.plot(indices, fes_md[:,1], color=color, lw=plot_linewidth, ls='-', alpha=theory_alpha, zorder=plot_zorder)
ax3.plot(indices, fes_md[:,2], color=color, lw=plot_linewidth, ls='-', alpha=theory_alpha, zorder=plot_zorder)


##########################################################################################################################################

params_model = 'cgnaplus'
hard_constraint = True
ncores = 4
verbose = False
use_correction = True

shl_open_left = 0
shl_open_right = 0

factors = [0.65,0.65,0.75,0.5,0.5,0.2]  
factors=[0.4220, 0.6619, 0.1556, 0.0626, 0.3414, 0.4118]
factors = None
factors=[0.52, 0.52, 0.62, 0.7, 0.7, 0.5]

nfe = rbpnfe.NucFreeEnergy(
    params_model = params_model,
    hardconstraint=hard_constraint,
    rescale_factors=factors
    )

t1 = time.time()
fes = nfe.eval_landscape(
    seq,
    shl_open_left = shl_open_left,
    shl_open_right = shl_open_right,
    use_correction = use_correction,
    ncores = ncores,
    verbose = verbose
    )
t2 = time.time()
print(f'Time elapsed {t2-t1:.4f} s') 

indices = np.arange(len(fes))
color = colors[-1]
label = 'CGNA+'
ax1.plot(indices, fes[:,0], color=color, lw=plot_linewidth, ls='-', alpha=theory_alpha, zorder=plot_zorder, label=label)
ax2.plot(indices, fes[:,1], color=color, lw=plot_linewidth, ls='-', alpha=theory_alpha, zorder=plot_zorder)
ax3.plot(indices, fes[:,2], color=color, lw=plot_linewidth, ls='-', alpha=theory_alpha, zorder=plot_zorder)

enth_shifted = fes[:,2] - np.mean(fes[:,2]) + np.mean(fes_md[:,2])
ax3.plot(indices, enth_shifted, color=color, lw=plot_linewidth, ls='-', alpha=0.7, zorder=plot_zorder)


ax1.legend(fontsize=legend_fontsize,borderpad=0.2,framealpha=0.8,fancybox=True,handlelength=0.8,handletextpad=0.5,loc='lower left', bbox_to_anchor=(-0.0075,-0.01),ncol=1,columnspacing=0.8)




ax1.set_ylabel(r'Free Energy ($\mathbf{k_B T}$)', fontsize=label_fontsize, 
                fontweight=label_fontweight, labelpad=0)
ax2.set_ylabel(r'Fluctuation Free Energy ($\mathbf{k_B T}$)', fontsize=label_fontsize, 
                fontweight=label_fontweight, labelpad=0)
ax3.set_ylabel(r'Enthalpy ($\mathbf{k_B T}$)', fontsize=label_fontsize, 
                fontweight=label_fontweight, labelpad=0)


ylabel_x_left = 0.09
xlabel_y = 0.09

for ax in axes:
    ax.yaxis.set_label_coords(-ylabel_x_left, 0.5 )
    ax.xaxis.set_label_coords(0.5, -xlabel_y )
    ax.set_xlim([indices[0],indices[-1]])
    ax.set_xlabel(r'Base Pair Index', fontsize=label_fontsize, 
                    fontweight=label_fontweight, labelpad=0)
# ax3.set_xlim([-2,102])



##########################################################################################################################################
##########################################################################################################################################
# Axes configs
for ax in axes:
    ###############################
    # set major and minor ticks
    ax.tick_params(axis="both",which='major',direction="in",width=axtick_major_width,length=axtick_major_length,labelsize=tick_labelsize,pad=tick_pad,color='#cccccc')
    ax.tick_params(axis='both',which='minor',direction="in",width=axtick_minor_width,length=axtick_minor_length,color='#cccccc')
    ax.minorticks_on()

    ###############################
    ax.xaxis.set_ticks_position('both')
    # set ticks right and top
    ax.yaxis.set_ticks_position('both')
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(axlinewidth)
        ax.spines[axis].set_color('grey')
        ax.spines[axis].set_alpha(0.7)

# savefn += f'_{metric}'

plt.subplots_adjust(left=0.12,
                    right=0.95,
                    bottom=0.06,
                    top=0.98,
                    wspace=0.5,
                    hspace=0.3)

savefn = f'figs/landscape_comparison_{nbp}bp'

fig.savefig(savefn+'.pdf',dpi=300,transparent=True)
fig.savefig(savefn+'.svg',dpi=300,transparent=True)
fig.savefig(savefn+'.png',dpi=300,transparent=False)
