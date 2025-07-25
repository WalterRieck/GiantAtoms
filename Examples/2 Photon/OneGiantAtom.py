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
nAtoms = 1
J = 1
g = 0.2
delta2 = np.sqrt(2)
delta1 = np.sqrt(2)
dx = 4
N = 100
CouplePoints = [[int((N + 1 + dx) / 2), int((N + 1 - dx) / 2)], [int((N + 1 + 2 * dx) / 2) , int((N + 1) / 2)]]
nT, maxT = 1000, 100

# Construct two photon GA object TGA and compute the eigenvalues/vectors of the Hamiltonian
giantAtom = TGA(N, J, nAtoms, [[g, g], [g, g]], [delta2], [delta1], CouplePoints)
es, vs = giantAtom.computeEigens()
ts = np.linspace(0, maxT, nT)

# Initial state with population in atom 1 and nothing in the cavities
initialState = giantAtom.constructInitialState(atoms=[1, 0], cavities=[0])

# Calculate the dynamics for level f, e, and ee
Ns2, Ns1, EE = giantAtom.computeDynamics2_0(ts, initialState)


# Plot the dynamics of the f and e levels
for i in range(nAtoms):
    plt.plot(ts, Ns2[i], label=r'$|f \rangle$', linewidth=3)
    plt.plot(ts, Ns1[i], label=r'$|e \rangle$', linewidth=3)
    plt.plot()

plt.title(r"Atom dynamics, $g/J = %.1f, dx=%i$" % (g, dx))
plt.xlabel(r'$tJ$')
plt.ylabel(r'$|C_e(t)|^2$')
plt.ylim(-0.1, 1.1)
plt.legend()
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 1, figsize=(7, 6))


# Calculate the site dynamics for the e, g and gg level. (gg level is the states with 2 photons in the same cavity)
resultE, resultG, resultGG = giantAtom.computeSiteDynamics(ts, initialState)
# Only interested in the e and g levels -> plotted below:
cax2 = axs.imshow(resultG.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.05), aspect='auto')
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
axs.set_title(r'$\Delta_2 = %.3f, \Delta_1$ = %.3f' % (delta2, delta1))

fig.suptitle('Wavepacket Dynamics, dx = %.i' % dx)
cbar2 = fig.colorbar(cax2, ax=axs, orientation='horizontal', location='bottom',
                     ticks=np.linspace(0, 0.05, 3))
cbar2.set_label(r'$P_G(n, t)$', labelpad=10)
fig.tight_layout()
fig.show()

fig, axs = plt.subplots(1, 1, figsize=(7, 6))

cax3 = axs.imshow(resultE.T, cmap='hot', norm=Normalize(vmin=0, vmax=0.25), aspect='auto')
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
axs.set_title(r'$\Delta_2 = %.3f, \Delta_1$ = %.3f' % (delta2, delta1))

fig.suptitle('Wavepacket Dynamics, dx = %.i' % dx)
cbar2 = fig.colorbar(cax3, ax=axs, orientation='horizontal', location='bottom',
                     ticks=np.linspace(0, 0.25, 3))
cbar2.set_label(r'$P_E(n, t)$', labelpad=10)
fig.tight_layout()
fig.show()


