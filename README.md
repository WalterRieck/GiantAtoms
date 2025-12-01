# Summary 

This is a python library for the simulation of Giant Atoms coupled to 1D structured environments.

# Installation

Clone the library with 
```bash
git clone https://github.com/WalterRieck/GiantAtoms
```

# Usage

Provided with the repository is a few examples in the "Examples" folder as well as the figures created from them in the "Figures" folder. One can run a particular example with
```bash
python Examples/Doublon/DirectCouplingGA.py
```
This will then generate the figures in: "Figures/Doublon/DirectCoupling".

# Classes

The following classes are available in the source file GA.py:

## GA / SA
Single photon giant atom and small atom creation.
```python
GA(N, J, nAtoms, gs, deltas, couplePoints)
SA(N, J, nAtoms, gs, deltas, couplePoints)
```
Create a giant or small atom coupled to a structured waveguide by specifying the size of the waveguide (N), the coupling between neighboring cavities (J), number of atoms (nAtoms), coupling strengths of the atom to the waveguide for each atom and couple point (gs), detuning of the atoms (deltas) and coupling point indices of each atom and coupling point (couplePoints). N and nAtoms are integers, J is a float, deltas is a 1D list [delta1, ...,  deltan] and couplePoints and gs are 2D lists [[g11, ..., g1n], ... , [gn1, gnn]]. 

## TGA / TSA
Two photon giant atom and small atom creation.
```python
TGA(N, J, nAtoms, gs, deltas2, deltas1, couplePoints)
TSA(N, J, nAtoms, gs, deltas2, deltas1, couplePoints)
```
Create a three level giant or small atom coupled to a structured waveguide by specifying the size of the waveguide (N), the coupling between neighboring cavities (J), number of atoms (nAtoms), coupling strengths of the atom to the waveguide for each atom and couple point (gs), detuning of the second level of the atoms (deltas2), detuning of the first level of the atoms (deltas1) and coupling point indices of each atom and coupling point (couplePoints). N and nAtoms are integers, J is a float, deltas1 and deltas2 are 1D lists [delta21, ...,  delta2n], [delta11, ..., delta1n] and couplePoints and gs are 2D lists [[g11, ..., g1n], ... , [gn1, gnn]]. 

## TGADoublon
Two photon giant atom with added kerr potential to two-photon cavities.
```python
TGADoublon(N, J, U, nAtoms, gDC, gs2, gs1, deltas2, deltas1, couplePoints)
```
Create a three level giant atom with an added kerr potential coupled to a structured waveguide by specifying the size of the waveguide (N), the coupling between neighboring cavities (J), kerr potential (U),  number of atoms (nAtoms), direct coupling strengths of the second level of the atom to the waveguide for each atom and couple point (gsDC), coupling strength between level 2 and level 1 of the atom (gs2), coupling strengths between the first level of the atom and the waveguide (gs1), detuning of the second level of the atoms (deltas2), the detuning of the first level of the atoms (deltas1) and the coupling point indices of each atom and coupling point (couplePoints). N and nAtoms are integers, J is a float, deltas1 and deltas2 are 1D lists [delta21, ...,  delta2n], [delta11, ..., delta1n] and couplePoints, gsDC, gs2, gs1  are 2D lists [[g11, ..., g1n], ... , [gn1, gnn]]. 
Here usually gs1 = gs2.

# Methods

The main methods associated with each of the above classes are:
```python
ConstructHamiltonian()
constructInitialState(atoms, cavities)
computeDynamics2_0(ts, initialState)
computeSiteDynamics(ts, initialState)
computeIPR()
saveGA(filename)
loadGA(filename)
```
## ConstructHamiltonian()

The ConstructHamiltonian method constructs the matrix Hamiltonian for the giant or small atom and can be accessed with self.Hamiltonian

## constructInitialState(atoms, cavities)

The constructInitialState method is used to obtain an initial state with all the population in a specified state. This state can be specified by supplying the atoms and cavities that should have population initially. One can for instance specify the initial state $| f_1, 00 \rangle$ by calling
```python
constructInitialState([1, 0], [0])
```
The numbering here starts at 1 and a 0 specifies that there is no population.

## computeDynamics2_0(ts, initialState)

This method takes an array of times ($t \approx 200$ is usually good) and an initial state and returns the population in $| f_i, 00 \rangle$, $\sum_j| e_i, j0 \rangle$ and $| ee, 00 \rangle$ as well as computes the states for all ts to be used later.

## computeSiteDynamics(ts, initialState)

The computeSiteDynamics method is used to compute the final population of all states with population in a cavity and returns three arrays corresponding to the populations in $\sum_i| ei, j0 \rangle$, $\sum_i| g, ij \rangle$ and $| g, jj \rangle$.

## computeIPR()

The computeIPR() method computes the Inverse Participation Ratio: $\sum_n |v(n)|^4$ for the eigenvalues v of the Hamiltonian. This can be used to find highly localized systems.

## saveGA(filename)

The SaveGA method can be used to save all information contained with the class. This can lead to the creation of very large files if simulations with a large number of ts is used.

## loadGA(filename)

The loadGA method can be used to load a previously saved class to for example run a longer simulation without having to calculate the eigenvalues again.
