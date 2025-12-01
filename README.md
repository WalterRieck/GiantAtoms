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

The following classes are available in the source file GA.py

## GA / SA
Single photon giant atom and small atom creation

## TGA / TSA
Two photon giant atom and small atom creation

## TGADoublon
Two photon giant atom with added kerr potential to two-photon cavities

# Examples

In the examples folder one can find examples for single photon, two photon and Doublon processes. For single photon and two photon processes we show dynamics for both a single GA as well as two braided GAs while for the doublon we only show dynamics for two giant atoms with direct or natural coupling. Running these examples yield the figures found in the corresponding folder in the Figures folder. 
