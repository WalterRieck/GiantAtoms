import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from matplotlib.colors import Normalize
from matplotlib.colors import PowerNorm
from src.GA import *
import seaborn as sns
import matplotlib as mpl


# Latex font
plt.rcParams['font.size'] = 20
plt.rcParams['figure.figsize'] = [7, 6]
mpl.rcParams.update({
    'text.usetex': True,
    'font.family': 'times',
    'text.latex.preamble': r'''
        \usepackage{times}
        \usepackage{amsmath,amssymb,bm}
        \DeclareSymbolFont{letters}{OML}{cmm}{m}{it}
    '''
})


# Settings
nAtoms = 2
J = 1
g = 0.2
delta = np.sqrt(2)
dx = 4
N = 400
CouplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)], [int((N + 1 + 2 * dx) / 2) , int((N + 1) / 2)]]
nT, maxT = 600, 100


# Create single photon GA object GA
giantAtom = GA(N, J, nAtoms, [[g, g], [g, g]], [delta, delta], CouplePoints)
es, vs = giantAtom.computeEigens()
ts = np.linspace(0, 100, 600)

# Initial state with photon in atom 1
initialState = giantAtom.constructInitialState(atom=1, cavity=0)

# Calculate dynamics for the e level
Ns = giantAtom.computeDynamics(ts, initialState)

# Plotting
iter = 1
for ns in Ns:
    plt.plot(ts, ns, label=r'Atom ' + str(iter), linewidth=3)
    iter += 1

plt.title(r"Atom dynamics, $g/J = %.1f, dx=%i$" % (g, dx))
plt.xlabel(r'$tJ$')
plt.ylabel(r'$|C_e(t)|^2$')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(7, 6))


# Calculate site dynamics for g level
result_n = giantAtom.computeWavepacketDynamics(ts, initialState)

# Plotting
cax2 = axs.imshow(result_n.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.1), aspect='auto')
axs.invert_yaxis()

numTicksY = 3
numTicksX = 6
tickPositionsY = np.linspace(0, N - 1, numTicksY, dtype=int)
tickLabelsY = np.linspace(0, N, numTicksY, dtype=int)
tickPositionsX = np.linspace(0, nT-1, numTicksX, dtype=int)
tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

axs.set_xticks(tickPositionsX)
axs.set_yticks(tickPositionsY)
axs.set_xticklabels(tickLabelsX)
axs.set_yticklabels(tickLabelsY)
axs.set_xlabel(r'$tJ$')
axs.set_ylabel(r'Cavity index $(n)$')
axs.set_title(r'$\Delta$ = %.3f' % delta)

fig.suptitle('Wavepacket Dynamics, dx = %.i' % dx)
cbar2 = fig.colorbar(cax2, ax=axs, orientation='horizontal', location='bottom',
                     ticks=np.linspace(0, 0.1, 3))
cbar2.set_label(r'$P(n, t)$', labelpad=10)
fig.tight_layout()
fig.show()







