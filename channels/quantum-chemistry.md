# Quantum chemistry

The energies of molecules from their electrons: Hartree-Fock, the exact answer in a finite basis, and excited states from linear response. Each post builds its method from the integrals up and tests every energy against PySCF.

## The hydrogen molecule
- h2-rhf-uhf-and-full-ci-along-the-bond — H2 in STO-3G: the correlation energy at equilibrium, RHF's wrong dissociation limit, and where UHF breaks away

## Excited states
- casida-tdhf-and-tda-for-water — water in 6-31G: all 40 singlet and 40 triplet excitation energies by TDHF and by the Tamm-Dancoff approximation, against PySCF
