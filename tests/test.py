import pytest
from qsci import qsciclass, vqe
from pyscf.fci import direct_spin1, FCI
import numpy as np
from pyscf import gto, scf, ao2mo
import pennylane as qml
from qsci.invham import InvHam

au2eV = 27.2114

def test_energy():
    norb = 2
    nelec = 2
    int1e = np.random.rand(norb, norb)
    int1e = int1e + int1e.T
    int2e = np.zeros((norb, norb, norb, norb))
    uccsd = vqe.UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = qsciclass.Sampler(uccsd)
    qscicls = qsciclass.QSCI(smp)
    qscicls.nchoose = 4
    e1, c1 = qscicls.diagonalize_sci()
    cis = direct_spin1.FCISolver()
    e2, c2 = cis.kernel(int1e, int2e, norb, nelec)
    assert abs(e1 - e2) < 1e-6
    return


def test_H2():
    norb = 2
    nelec = 2
    mol = gto.Mole()
    mol.atom = """H 0 0 0; H 0 0 1.0"""
    mol.basis = "sto-3g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore = mf.get_hcore()
    int1e = np.einsum("ia,jb,ij->ab", mf.mo_coeff, mf.mo_coeff, hcore)
    int2e = ao2mo.kernel(mol, mf.mo_coeff)
    int2e = ao2mo.addons.restore("s1", int2e, mf.mo_coeff.shape[1])
    print(int2e.shape)
    uccsd = vqe.UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = qsciclass.Sampler(uccsd)
    qscicls = qsciclass.QSCI(smp)
    qscicls.nchoose = 2
    e1, c1 = qscicls.diagonalize_sci()
    cis = direct_spin1.FCISolver()
    e2, c2 = cis.kernel(int1e, int2e, norb, nelec)
    print(c2.shape)
    assert abs(e1 - e2) < 1e-6
    return


def test_H4():
    norb = 4
    nelec = 4
    mol = gto.Mole()
    mol.atom = """H 0 0 0; H 0 0 0.5; H 0 0 1.0; H 0 0 1.5"""
    mol.basis = "sto-3g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore = mf.get_hcore()
    int1e = np.einsum("ia,jb,ij->ab", mf.mo_coeff, mf.mo_coeff, hcore)
    int2e = ao2mo.kernel(mol, mf.mo_coeff)
    int2e = ao2mo.addons.restore("s1", int2e, mf.mo_coeff.shape[1])
    print(int2e.shape)
    uccsd = vqe.UCCSD_Lattice(int1e, int2e, norb, nelec)
    uccsd.optimize()
    smp = qsciclass.Sampler(uccsd)
    qscicls = qsciclass.QSCI(smp)
    qscicls.nchoose = 10
    e1, c1 = qscicls.diagonalize_sci()
    cis = direct_spin1.FCISolver()
    e2, c2 = cis.kernel(int1e, int2e, norb, nelec)
    print(c2.shape)
    # chemical accuracy
    assert abs(e1 * au2eV - e2 * au2eV) < 0.03
    return


def test_H2_inv_ham():
    mol = gto.Mole()
    mol.atom = """H 0.01076341 0.04449877 0.0; H 0.98729513 1.63059094 0.0"""
    mol.unit = "B"
    mol.basis = "sto-3g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    hcore = mf.get_hcore()
    int1e = np.einsum("ia,jb,ij->ab", mf.mo_coeff, mf.mo_coeff, hcore)
    int2e = ao2mo.kernel(mol, mf.mo_coeff)
    cis = direct_spin1.FCISolver()
    e2, c2 = cis.kernel(int1e, int2e, 2, 2)
    e2 += mol.energy_nuc()
    print("e2 =", e2)

    symbols = ["H", "H"]
    geometry = np.array([[0.01076341, 0.04449877, 0.0], [0.98729513, 1.63059094, 0.0]])
    # Build the electronic Hamiltonian
    H, qubits = qml.qchem.molecular_hamiltonian(symbols, geometry)
    print(qubits)
    dev = qml.device("default.qubit", wires=qubits)
    ih = InvHam(H, qubits)

    @qml.qnode(dev)
    def hartree_fock_state():
        qml.BasisState(np.array([0, 1]), wires=[0, 1])  # |01⟩ 状態
        qml.RX(np.pi / 2, wires=0)  # 回転ゲートを使った初期化
        qml.RY(np.pi / 4, wires=1)
        return qml.state()

    s = hartree_fock_state()
    # s = qnp.random.rand(qubits**2)
    # s /= qnp.linalg.norm(s)
    E0 = ih.weighted_inv(s)
    
    ### transform to eV, chemical accuracy
    assert(abs(E0.item()*27.2114 - e2*27.2114) < 0.03)
    return

