from qsci.vqe import UCCSD_Lattice
from qsci.qsciclass import QSCI, Sampler
import numpy as np
from pyscf import ao2mo
from pyscf import gto, scf
from pyscf.tools import fcidump
from qsci.dmrg import DMRGCalculator

class QSCI_MPS:
    def __init__(self, mf):
        self.mf = mf
        self.scivec = None
        self.fcivec = None
        self.mps = None
        self.mpo = None
        self.energy = None
        self.mps_save = None
        self.int1e = None
        self.int2e = None
        self.norb = mf.mo_coeff.shape[1]
        self.nelec = mf.mol.nelectron
        self.ncore = 0  # Number of core orbitals, can be set later
        self.bond_dim = 50  # Default bond dimension, can be adjusted later
        self.dmrg = None
        self.mps = None
        self.uccsd = None
        self.max_steps = 10  # Default maximum steps for DMRG, can be adjusted later

    def get_init_guess(self, nuc=0.):
        """Perform a main step of the DMRG calculation."""
        e = self.run_qsci(0, nuc=nuc)
        print("QSCI energy at step 0:", e)
        self.dmrg.scivec = self.scivec
        mps = self.dmrg.run(do_dmrg=False)
        self.mps = mps
        return

    def kernel(self, max_steps=10, use_dmrg2=True, nuc=0.):
        # self.localize()
        self.max_steps = max_steps
        self.get_ints()
        self.dmrg = DMRGCalculator(
            self.mf,
            self.int1e,
            self.int2e,
            self.norb,
            self.nelec,
            ncore=self.ncore,
            bond_dim=self.bond_dim,
        )
        # self.mps = self.dmrg.hamil.build_mps(bond_dim=self.bond_dim)
        self.get_init_guess(nuc=nuc)
        self.dmrg.mps = self.mps
        print("Running DMRG...")
        self.dmrg.run(do_dmrg=True, nuc=nuc)
        self.energy = self.dmrg.energy
        # print("Final DMRG energy:", self.energy)
        return self.energy

    def get_ints(self):
        hcore = self.mf.get_hcore()
        self.int1e = np.einsum(
            "ia,jb,ij->ab", self.mf.mo_coeff, self.mf.mo_coeff, hcore
        )
        self.int2e = ao2mo.kernel(self.mf.mol, self.mf.mo_coeff)
        self.int2e = ao2mo.addons.restore("s1", self.int2e, self.mf.mo_coeff.shape[1])
        return

    def localize(self):
        """Localize the orbitals using the PySCF localization method."""
        from pyscf import lo

        self.mf.mo_coeff = lo.orth.orth_ao(self.mf.mol, "lowdin")
        return

    def run_qsci(self, step, nchoose=30, nuc=0.):
        # print("norb, nelec, scivec @run_qsci:", self.norb, self.nelec)
        if step == 0:
            uccsd = UCCSD_Lattice(self.int1e, self.int2e, self.norb, self.nelec)
            uccsd.optimize()
            self.uccsd = uccsd
        else:
            uccsd = self.uccsd
        smp = Sampler(uccsd)
        qscicls = QSCI(smp)
        qscicls.nchoose = nchoose
        e1, c1 = qscicls.diagonalize_sci()
        self.scivec = c1
        return e1 + nuc
