from src.GA import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
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


# Calculates the decoherence-free frequency inside the doublon band
def DFIfreq(k, U, J):
    return np.sqrt(U ** 2 + 16 * J ** 2 * np.cos(k / 2) ** 2)

# Settings
U = 10
J = 1
nAtoms = 2
N = 100
dx = 2
gs = [[0.25, 0.25], [0.25, 0.25]]    # Natural coupling
gsDC = [[0.0, 0.0], [0.0, 0.0]]      # Direct coupling
couplingPoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)], [int((N + 1 + 2 * dx) / 2) , int((N + 1) / 2)]]
maxT, nT = 1000, 1000
ts = np.linspace(0, maxT, nT)
k = np.pi / dx
delta2 = [5.0, 5.0]
delta1 = [5.338, 5.338]


# Construct TGADoublon object
giantAtom = TGADoublon(N, J, U, nAtoms, gsDC, gs, gs, delta2, delta1, couplingPoints)

# Initial state with population in atom 1 and in no cavity i.e |fg, 0>
initialState1 = giantAtom.constructInitialState(atoms=[1, 0], cavities=[0])

# Computes dynamics for level f, e and the |ee, 0> state
Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState1)

# Plot level f since it is the only level that is expected to have significant popualtion due to DFI
iter = 1
for ns2 in Ns2:
    plt.plot(ts, ns2, linewidth=4, label=r'$\Delta_2 = $ %.3f' % delta2[0])
    iter += 1


plt.xlabel(r'$tJ$')
plt.ylabel(r'$n^{(2)}(t)$')
plt.title(r'$\Delta_1 = %.3f$' % delta1[0])
plt.ylim([-0.1, 1.1])
plt.legend()
plt.tight_layout()
plt.show()


# Compute the sitedynamics for level e, g and for the doublon states gg
heatmapE, heatmapG, heatmapGG = giantAtom.computeSiteDynamics(ts, initialState1)

maxPop = heatmapGG.max()
fig, ax = plt.subplots(figsize=(7, 6))


# Since we are in the doublon band we only expect population in the gg states
cax2 = ax.imshow(heatmapGG.T, cmap='hot', norm=Normalize(vmin=0, vmax=maxPop), aspect='auto')

numTicksY = 3
numTicksX = 3
tickPositionsY = np.linspace(0, N-1, numTicksY, dtype=int)
tickLabelsY = np.linspace(1, N, numTicksY, dtype=int)
tickPositionsX = np.linspace(0, nT - 1, numTicksX, dtype=int)
tickLabelsX1 = ['0', '2.5e2', '5.0e2']
tickLabelsX = np.linspace(0, maxT, numTicksX, dtype=int)

ax.set_xticks(tickPositionsX, tickLabelsX)
ax.set_yticks(tickPositionsY, tickLabelsY)
ax.set_xlabel(r'$t\;J$')
ax.set_ylabel(r'Cavity index $(i)$')

cbar2 = fig.colorbar(cax2, ax=ax, orientation='horizontal', pad=0.2)
fig.tight_layout()
fig.show()

