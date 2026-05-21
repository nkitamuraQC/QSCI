from qsci.qsci_mps import QSCI_MPS
from pyscf import gto, scf, ao2mo, fci
import numpy as np

au2eV = 27.2114


def test_H4():
    mol = gto.Mole()
    mol.atom = """H 0 0 0; H 0 0 0.5; H 0 0 1.0; H 0 0 1.5; H 0 0 2.0; H 0 0 2.5"""
    # mol.atom = """H 0 0 0; H 0 0 0.5; H 0 0 1.0; H 0 0 1.5;"""
    mol.basis = "sto-3g"
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()
    nuc = mf.energy_nuc()
    hcore = mf.get_hcore()
    int1e = np.einsum(
        "ia,jb,ij->ab", mf.mo_coeff, mf.mo_coeff, hcore
    )
    int2e = ao2mo.kernel(mf.mol, mf.mo_coeff)
    int2e = ao2mo.addons.restore("s1", int2e, mf.mo_coeff.shape[1])
    cis = fci.direct_spin1.FCISolver(mol)
    e, c = cis.kernel(int1e, int2e, mf.mo_coeff.shape[1], mf.mol.nelectron, )
    print(e+nuc)

    qsci_mps_cls = QSCI_MPS(mf)
    qsci_mps_cls.get_ints()
    qsci_mps_cls.kernel(nuc=nuc)
    # print("DMRG energy =", energy)
    return


if __name__ == "__main__":
    test_H4()
