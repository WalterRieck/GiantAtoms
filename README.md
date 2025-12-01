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

# Functions

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
