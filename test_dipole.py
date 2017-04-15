from __future__ import print_function
from __future__ import division

import numpy as np
import numpy.linalg as npl

import pyquante2
from pyquante2.geo.molecule import molecule
from pyints.one import makeM

from dipole import (nuclear_dipole_contribution,
                    nuclear_dipole_contribution_pyquante,
                    electronic_dipole_contribution,
                    get_isotopic_masses, calc_center_of_mass_pyquante,
                    calc_center_of_mass,
                    calc_center_of_nuclear_charge,
                    calc_center_of_electronic_charge_pyquante,
                    calculate_origin, calculate_dipole)


def test_dipole_LiH_H2_HF_STO_3G():

    #   X      -4.8174      Y       0.9597      Z      -0.0032
    # Tot       4.9121
    qchem_total_components_debye = np.array([-4.8174, 0.9597, -0.0032])
    qchem_total_norm_debye = 4.9121

    #      DX          DY          DZ         /D/  (DEBYE)
    # -4.817430    0.959709   -0.003226    4.912096
    gamess_total_components_debye = np.array([-4.817430, 0.959709, -0.003226])
    gamess_total_norm_debye = 4.912096

    #                                Dipole moment
    #                                -------------

    #               au               Debye          C m (/(10**-30)
    #            1.932564           4.912086          16.384956


    #                           Dipole moment components
    #                           ------------------------

    #               au               Debye          C m (/(10**-30)

    #    x     -1.89531979        -4.81742209       -16.06919039
    #    y      0.37757524         0.95970046         3.20121617
    #    z     -0.00126928        -0.00322619        -0.01076141


    # Units:   1 a.u. =   2.54175 Debye
    #          1 a.u. =   8.47835 (10**-30) C m (SI)
    dalton_total_components_debye = np.array([-4.81742209, 0.95970046, -0.00322619])
    dalton_total_norm_debye = 4.912086
    dalton_total_components_au = np.array([-1.89531979, 0.37757524, -0.00126928])
    dalton_total_norm_au = 1.932564
    dalton_center_of_mass_au = np.array([-2.468120057069, 2.168586684080, -0.007311931664])

    # ORCA uses the center of mass?

    # Electronic contribution:     -4.65190      -3.56492       0.02433
    # Nuclear contribution   :      2.75658       3.94249      -0.02560
    #                         -----------------------------------------
    # Total Dipole Moment    :     -1.89532       0.37758      -0.00127
    #                         -----------------------------------------
    # Magnitude (a.u.)       :      1.93256
    # Magnitude (Debye)      :      4.91219
    orca_electronic_components_au = np.array([-4.65190, -3.56492,  0.02433])
    orca_nuclear_components_au = np.array([2.75658, 3.94249, -0.02560])
    orca_total_components_au = np.array([-1.89532, 0.37758, -0.00127])
    assert (((orca_nuclear_components_au + orca_electronic_components_au) - orca_total_components_au) < 1.0e-14).all()
    orca_total_norm_au = 1.93256
    assert abs(orca_total_norm_au - npl.norm(orca_total_components_au)) < 1.0e-5
    orca_total_norm_debye = 4.91219

    # Origin is the Cartesian origin
    # Nuclear Dipole Moment: (a.u.)
    #    X:   -12.0198      Y:    17.0002      Z:    -0.0698

    # Electronic Dipole Moment: (a.u.)
    #    X:    10.1245      Y:   -16.6226      Z:     0.0685

    # Dipole Moment: (a.u.)
    #    X:    -1.8953      Y:     0.3776      Z:    -0.0013     Total:     1.9326

    # Dipole Moment: (Debye)
    #    X:    -4.8174      Y:     0.9597      Z:    -0.0032     Total:     4.9121
    psi4_nuclear_components_au = np.array([-12.0198, 17.0002, -0.0698])
    psi4_electronic_components_au = np.array([10.1245, -16.6226, 0.0685])
    psi4_total_components_au = np.array([-1.8953, 0.3776, -0.0013])
    assert (((psi4_nuclear_components_au + psi4_electronic_components_au) - psi4_total_components_au) < 1.0e-14).all()
    psi4_total_norm_au = 1.9326
    assert abs(psi4_total_norm_au - npl.norm(psi4_total_components_au)) < 1.0e-4
    psi4_total_components_debye = np.array([-4.8174, 0.9597, -0.0032])
    psi4_total_norm_debye = 4.9121
    assert abs(psi4_total_norm_debye - npl.norm(psi4_total_components_debye)) < 1.0e-4

    mol = molecule(
        [
            (3,        -1.67861,        0.61476,       -0.00041),
            (1,        -0.01729,        0.38654,       -0.00063),
            (1,        -0.84551,        3.08551,       -0.00236),
            (1,        -0.46199,        3.67980,       -0.03270)
        ],
        units='Angstrom',
        charge=0,
        multiplicity=1,
        name='LiH_H2')

    nuccoords = np.array([atom.r for atom in mol.atoms])
    nuccharges = np.array([atom.Z for atom in mol.atoms])[..., np.newaxis]
    masses = get_isotopic_masses(nuccharges[:, 0])

    mol_basis = pyquante2.basisset(mol, 'STO-3G'.lower())

    solver = pyquante2.rhf(mol, mol_basis)
    solver.converge(tol=1e-11, maxiters=1000)

    # print(solver)

    C = solver.orbs
    NOa = mol.nup()
    NOb = mol.ndown()
    assert NOa == NOb
    D = 2 * np.dot(C[:, :NOa], C[:, :NOa].T)

    origin_zero = np.array([0.0, 0.0, 0.0])

    ref = psi4_nuclear_components_au
    res = nuclear_dipole_contribution(nuccoords, nuccharges, origin_zero)
    abs_diff = np.absolute(ref - res)
    assert (abs_diff < 1.0e-4).all()

    ref = psi4_electronic_components_au
    res = electronic_dipole_contribution(D, mol_basis, origin_zero)
    abs_diff = np.absolute(ref - res)
    assert (abs_diff < 1.0e-4).all()

    res1 = nuclear_dipole_contribution(nuccoords, nuccharges, origin_zero)
    res2 = nuclear_dipole_contribution_pyquante(mol, origin_zero)
    assert ((res1 - res2) < 1.0e-15).all()

    ref = dalton_center_of_mass_au
    res = calc_center_of_mass_pyquante(mol)
    abs_diff = np.absolute(ref - res)
    assert (abs_diff < 1.0e-6).all()
    com = res

    res1 = calc_center_of_mass_pyquante(mol)
    res2 = calc_center_of_mass(nuccoords, masses)
    assert ((res1 - res2) < 1.0e-15).all()

    ncc = calc_center_of_nuclear_charge(nuccoords, nuccharges)
    assert ((ncc - np.array([-2.00330482, 2.83337011, -0.01162811])) < 1.0e-8).all()
    ecc = calc_center_of_electronic_charge_pyquante(D, mol_basis)
    assert ((ecc - np.array([1.68741793, -2.77044101, 0.01141657])) < 1.0e-8).all()

    origin_zero = calculate_origin('zero', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_zero = calculate_dipole(nuccoords, nuccharges, origin_zero, D, mol_basis, do_print=True)
    origin_com = calculate_origin('com', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_com = calculate_dipole(nuccoords, nuccharges, origin_com, D, mol_basis, do_print=True)
    origin_ncc = calculate_origin('ncc', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_ncc = calculate_dipole(nuccoords, nuccharges, origin_ncc, D, mol_basis, do_print=True)
    origin_ecc = calculate_origin('ecc', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_ecc = calculate_dipole(nuccoords, nuccharges, origin_ecc, D, mol_basis, do_print=True)

    # For an uncharged system, these should all be identical.
    my_ref = np.array([-1.89532134e+00, 3.77574623e-01, -1.26926571e-03])
    for res in (dipole_zero, dipole_com, dipole_ncc, dipole_ecc):
        assert (np.absolute(my_ref - res) < 1.0e-8).all()

    return


def test_dipole_hydroxyl_radical_HF_STO_3G():

    qchem_final_energy = -74.3626375184
    # Dipole Moment (Debye)
    #      X       0.0000      Y      -0.0000      Z      -1.2788
    #    Tot       1.2788
    qchem_total_components_debye = np.array([0.0000, 0.0000, -1.2788])
    qchem_total_norm_debye = 1.2788

    dalton_final_energy = -74.361530725817
    #                             Dipole moment
    #                             -------------

    #            au               Debye          C m (/(10**-30)
    #         0.502283           1.276676           4.258534


    #                        Dipole moment components
    #                        ------------------------

    #            au               Debye          C m (/(10**-30)

    # x     -0.00000000        -0.00000000        -0.00000000
    # y     -0.00000000        -0.00000000        -0.00000000
    # z     -0.50228316        -1.27667636        -4.25853394

    # Units:   1 a.u. =   2.54175 Debye
    #          1 a.u. =   8.47835 (10**-30) C m (SI)
    dalton_total_components_debye = np.array([0.0, 0.0, -1.27667636])
    dalton_total_norm_debye = 1.276676
    dalton_total_components_au = np.array([0.0, 0.0, -0.50228316])
    dalton_total_norm_au = 0.502283
    dalton_center_of_mass_au = np.array([0.0, 0.0, 1.723849254747])

    # ORCA uses the center of mass?
    orca_final_energy = -74.362637379044
    # Electronic contribution:      0.00000      -0.00000       0.35185
    # Nuclear contribution   :      0.00000       0.00000      -0.85498
    #                         -----------------------------------------
    # Total Dipole Moment    :      0.00000      -0.00000      -0.50312
    #                         -----------------------------------------
    # Magnitude (a.u.)       :      0.50312
    # Magnitude (Debye)      :      1.27884
    orca_electronic_components_au = np.array([0.0, 0.0, 0.35185])
    orca_nuclear_components_au = np.array([0.0, 0.0, -0.85498])
    orca_total_components_au = np.array([0.0, 0.0, -0.50312])
    assert (((orca_nuclear_components_au + orca_electronic_components_au) - orca_total_components_au) < 1.0e-14).all()
    orca_total_norm_au = 0.50312
    assert abs(orca_total_norm_au - npl.norm(orca_total_components_au)) < 1.0e-5
    orca_total_norm_debye = 1.27884

    psi4_final_energy = -74.3626375190713986
    # Origin is the Cartesian origin
    # Nuclear Dipole Moment: (a.u.)
    #    X:     0.0000      Y:     0.0000      Z:    14.6597

    # Electronic Dipole Moment: (a.u.)
    #    X:    -0.0000      Y:     0.0000      Z:   -15.1629

    # Dipole Moment: (a.u.)
    #    X:    -0.0000      Y:     0.0000      Z:    -0.5031     Total:     0.5031

    # Dipole Moment: (Debye)
    #    X:    -0.0000      Y:     0.0000      Z:    -1.2788     Total:     1.2788
    psi4_nuclear_components_au = np.array([0.0, 0.0, 14.6597])
    psi4_electronic_components_au = np.array([0.0, 0.0, -15.1629])
    psi4_total_components_au = np.array([0.0, 0.0, -0.5031])
    assert (((psi4_nuclear_components_au + psi4_electronic_components_au) - psi4_total_components_au) < 1.0e-14).all()
    psi4_total_norm_au = 0.5031
    assert abs(psi4_total_norm_au - npl.norm(psi4_total_components_au)) < 1.0e-4
    psi4_total_components_debye = np.array([0.0, 0.0, -1.2788])
    psi4_total_norm_debye = 1.2788
    assert abs(psi4_total_norm_debye - npl.norm(psi4_total_components_debye)) < 1.0e-4

    mol = molecule([(1, 0.000, 0.000, 0.000),
                    (8, 0.000, 0.000, 0.9697)],
                   units='Angstrom',
                   charge=0,
                   multiplicity=2,
                   name='hydroxyl_radical')

    mol_basis = pyquante2.basisset(mol, 'STO-3G'.lower())

    solver = pyquante2.uhf(mol, mol_basis)
    solver.converge(tol=1e-11, maxiters=1000)

    C_alph = solver.orbsa
    C_beta = solver.orbsb
    NOa = mol.nup()
    NOb = mol.ndown()
    D_alph = np.dot(C_alph[:, :NOa], C_alph[:, :NOa].T)
    D_beta = np.dot(C_beta[:, :NOb], C_beta[:, :NOb].T)
    D = D_alph + D_beta

    nuccoords = np.array([atom.r for atom in mol.atoms])
    nuccharges = np.array([atom.Z for atom in mol.atoms])[..., np.newaxis]
    masses = get_isotopic_masses(nuccharges[:, 0])

    origin_zero = np.array([0.0, 0.0, 0.0])

    ref = psi4_nuclear_components_au
    res = nuclear_dipole_contribution(nuccoords, nuccharges, origin_zero)
    abs_diff = np.absolute(ref - res)
    assert (abs_diff < 1.0e-4).all()

    ref = psi4_electronic_components_au
    res = electronic_dipole_contribution(D, mol_basis, origin_zero)
    abs_diff = np.absolute(ref - res)
    assert (abs_diff < 1.0e-4).all()

    res1 = nuclear_dipole_contribution(nuccoords, nuccharges, origin_zero)
    res2 = nuclear_dipole_contribution_pyquante(mol, origin_zero)
    assert ((res1 - res2) < 1.0e-15).all()

    ref = dalton_center_of_mass_au
    res = calc_center_of_mass_pyquante(mol)
    abs_diff = np.absolute(ref - res)
    assert (abs_diff < 1.0e-6).all()
    com = res

    res1 = calc_center_of_mass_pyquante(mol)
    res2 = calc_center_of_mass(nuccoords, masses)
    assert ((res1 - res2) < 1.0e-15).all()

    ncc = calc_center_of_nuclear_charge(nuccoords, nuccharges)
    assert ((ncc - np.array([0.0, 0.0, 1.62885981])) < 1.0e-8).all()
    ecc = calc_center_of_electronic_charge_pyquante(D, mol_basis)
    assert ((ecc - np.array([0.0, 0.0, -1.68476254])) < 1.0e-8).all()

    origin_zero = calculate_origin('zero', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_zero = calculate_dipole(nuccoords, nuccharges, origin_zero, D, mol_basis, do_print=True)
    origin_com = calculate_origin('com', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_com = calculate_dipole(nuccoords, nuccharges, origin_com, D, mol_basis, do_print=True)
    origin_ncc = calculate_origin('ncc', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_ncc = calculate_dipole(nuccoords, nuccharges, origin_ncc, D, mol_basis, do_print=True)
    origin_ecc = calculate_origin('ecc', nuccoords, nuccharges, D, mol_basis, do_print=True)
    dipole_ecc = calculate_dipole(nuccoords, nuccharges, origin_ecc, D, mol_basis, do_print=True)

    # For an uncharged system, these should all be identical.
    my_ref = np.array([0.0, 0.0, -0.5031245309396919])
    for res in (dipole_zero, dipole_com, dipole_ncc, dipole_ecc):
        assert (np.absolute(my_ref - res) < 1.0e-8).all()

    return

if __name__ == '__main__':
    test_dipole_LiH_H2_HF_STO_3G()
    test_dipole_hydroxyl_radical_HF_STO_3G()
    pass
