#!/usr/bin/env python3

import matplotlib as mpl
from matplotlib import cm
# mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('bmh')

fig, ax = plt.subplots(1, 1, figsize=(1, 8))
fig.subplots_adjust(bottom=0.05,top=0.95,left=0,right=0.65)
norm = mpl.colors.Normalize(vmin=0., vmax=1.)
cb1 = mpl.colorbar.ColorbarBase(ax, cmap=cm.bwr,
                                norm=norm,
                                orientation='vertical', alpha=0.5)

print("Created legend.{pdf,png}")
plt.savefig('legend.pdf')
plt.savefig('legend.png')
