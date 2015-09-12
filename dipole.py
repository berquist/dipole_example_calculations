#!/usr/bin/env python2

from __future__ import print_function

import numpy as np
import numpy.linalg as npl

import pyquante2
from pyquante2.geo.molecule import molecule
from pyints.one import makeM


def calc_center_of_mass(mol):
    denominator = sum(atom.mass() for atom in mol.atoms)
    numerator = sum(atom.r * atom.mass() for atom in mol.atoms)
    return numerator / denominator


def nuclear_dipole_contribution(mol, origin_in_bohrs):
    return sum(atom.Z * (atom.r - origin_in_bohrs) for atom in mol.atoms)


def main(debug=False):

    mol = molecule([(1, 0.000, 0.000, 0.000),
                    (8, 0.000, 0.000, 0.9697)],
                   units='Angstrom',
                   charge=0,
                   multiplicity=2,
                   name='hydroxyl_radical')

    mol_basis = pyquante2.basisset(mol, 'STO-3G')

    solver = pyquante2.uhf(mol, mol_basis)
    solver.converge(tol=1e-11, maxiters=1000)

    print(solver)

    C_alph = solver.orbsa
    C_beta = solver.orbsb
    NOa = mol.nup()
    NOb = mol.ndown()
    D_alph = np.dot(C_alph[:, :NOa], C_alph[:, :NOa].transpose())
    D_beta = np.dot(C_beta[:, :NOb], C_beta[:, :NOb].transpose())
    D = D_alph + D_beta

    if debug:
        print(D_alph)
        print(D_beta)

    # This is origin used for the multipole analysis.
    # origin = np.array([0.0, 0.0, 0.0])
    origin = calc_center_of_mass(mol)
    print('Origin used: ({}, {}, {})'.format(*origin))

    M100_AO = makeM(mol_basis.bfs, origin, [1, 0, 0])
    M010_AO = makeM(mol_basis.bfs, origin, [0, 1, 0])
    M001_AO = makeM(mol_basis.bfs, origin, [0, 0, 1])

    M100_MO = D * M100_AO
    M010_MO = D * M010_AO
    M001_MO = D * M001_AO

    if debug:
        print('M100_AO')
        print(M100_AO)
        print('M010_AO')
        print(M010_AO)
        print('M001_AO')
        print(M001_AO)
        print('M100_MO')
        print(M100_MO)
        print('M010_MO')
        print(M010_MO)
        print('M001_MO')
        print(M001_MO)

    dipole_electronic_atomic_units = -np.array([np.sum(M100_MO), np.sum(M010_MO), np.sum(M001_MO)])

    convfac_au_to_debye = 2.541746230211

    dipole_nuclear_atomic_units = nuclear_dipole_contribution(mol, origin)
    dipole_total_atomic_units = dipole_nuclear_atomic_units + dipole_electronic_atomic_units

    dipole_magnitude_atomic_units = npl.norm(dipole_total_atomic_units)
    dipole_magnitude_debye = convfac_au_to_debye * dipole_magnitude_atomic_units

    print('=============')
    print('Dipole')
    print('=============')
    print('electronic (a.u.) {:8.5f} {:8.5f} {:8.5f}'.format(*dipole_electronic_atomic_units))
    print('   nuclear (a.u.) {:8.5f} {:8.5f} {:8.5f}'.format(*dipole_nuclear_atomic_units))
    print('     total (a.u.) {:8.5f} {:8.5f} {:8.5f}'.format(*dipole_total_atomic_units))
    print('Dipole moment magnitude')
    print(' {:8.5f} a.u'.format(dipole_magnitude_atomic_units))
    print(' {:8.5f} D'.format(dipole_magnitude_debye))

    print('=============')
    print('Origins')
    print('=============')
    print('             center of mass: {:f} {:f} {:f}'.format(*calc_center_of_mass(mol)))
    # print('center of electronic charge: {:f} {:f} {:f}'.format(*calc_center_of_electronic_charge()))


if __name__ == '__main__':
    main(debug=True)
