from pyscf.fci.selected_ci import to_fci
from pyscf.fci.cistring import addrs2str
import numpy as np
from scipy.sparse import coo_matrix
try:
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.algebra.mpe import MPE
except ImportError:
    print("pyblock3 not found, DMRG3 functionality will be unavailable.")

from pyblock2.driver.core import (
    DMRGDriver,
    SymmetryTypes,
)
from pyscf.tools import fcidump

import numpy as np

def sci_to_block2_mps(dets, coeffs, n_sites, n_elec, driver):
    """
    dets: shape (ndet, n_sites), values in {0,1,2,3}
    coeffs: shape (ndet,)
    """

    # 1. safety check
    dets = np.asarray(dets, dtype=np.uint8)
    coeffs = np.asarray(coeffs, dtype=np.float64)

    assert dets.shape[1] == n_sites, "n_sites mismatch"
    assert dets.shape[0] == coeffs.shape[0], "ndet mismatch"

    # 2. initialize driver

    driver.initialize_system(
        n_sites=n_sites,
        n_elec=n_elec,
    )

    # 3. build MPS from determinants (IMPORTANT)
    mps = driver.get_mps_from_csf_coefficients(
        dets,
        coeffs,
        tag="SCI_INIT",
        dot=2,
        full_fci=True,
    )

    return driver, mps

def fcidict_to_block2_dets(fcidict, n_sites, n_core=0):
    dets = []
    coeffs = []

    for (a_int, b_int), c in fcidict.items():

        det = np.zeros(n_sites, dtype=np.uint8)

        # core orbitals are assumed doubly occupied
        for i in range(n_core):
            det[i] = 3

        # active space
        for i in range(n_sites - n_core):
            ai = (a_int >> i) & 1
            bi = (b_int >> i) & 1

            det[i + n_core] = ai + 2 * bi  # 0/1/2/3 encoding

        dets.append(det)
        coeffs.append(c)

    return np.array(dets, dtype=np.uint8), np.array(coeffs, dtype=np.float64)


class DMRGCalculator:
    def __init__(
        self, mf, int1e, int2e, norb, nelec, scivec=None, ncore=0, bond_dim=10
    ):
        self.mf = mf
        self.int1e = int1e
        self.int2e = int2e
        self.norb = norb
        self.nelec = nelec
        self.ncore = ncore
        self.bond_dim = bond_dim
        self.scivec = scivec
        self.fcivec = None
        self.mps = None
        self.mpo = None
        self.dmrg2 = DMRGDriver(
            scratch="./tmp", symm_type=SymmetryTypes.SZ, n_threads=1
        )
        self.dmrg3 = None
        self.energy = None
        self.mps_save = None
        self.use_dmrg2 = False
        # self.get_mpo3()

    def to_fci(self):
        """Convert the selected CI vector to FCI format."""
        if self.scivec is None:
            raise ValueError("scivec must be provided to convert to FCI.")
        # print("norb, nelec, scivec @to_fci:", self.norb, self.nelec)
        self.fcivec = to_fci(self.scivec, self.norb, self.nelec)
        return

    def _get_fcivec_dict(self, tol=1e-10):
        if self.fcivec is None:
            raise ValueError(
                "fcivec must be provided to get the FCI vector dictionary."
            )
        self.fcivec[abs(self.fcivec) < tol] = 0
        sparse_fcimatr = coo_matrix(
            self.fcivec, shape=np.shape(self.fcivec), dtype=float
        )
        row, col, dat = sparse_fcimatr.row, sparse_fcimatr.col, sparse_fcimatr.data
        ncas_a, ncas_b = self.norb, self.norb
        nelecas_a, nelecas_b = self.nelec // 2, self.nelec // 2

        ## turn FCI wavefunction matrix indices into integers representing Fock occupation vectors
        ints_row = addrs2str(ncas_a, nelecas_a, row)
        ints_col = addrs2str(ncas_b, nelecas_b, col)

        ncore_a, ncore_b = self.ncore, self.ncore

        ## pad the integers to recover the full-space wavefunction
        padded_ints_row = (ints_row << ncore_a) | int(2**ncore_a - 1)
        padded_ints_col = (ints_col << ncore_b) | int(2**ncore_b - 1)

        ## create the FCI matrix as a dict
        dict_fcimatr = dict(zip(list(zip(padded_ints_row, padded_ints_col)), dat))
        self.fcidict = dict_fcimatr

        return

    def scivec2mps(self):
        dets, coeffs = fcidict_to_block2_dets(self.fcidict, self.norb, n_core=self.ncore)
        _, mps = sci_to_block2_mps(dets, coeffs, self.norb, self.nelec, self.dmrg2)
        return mps

    def get_mpo2(self):
        # fcidump.from_integrals(
        #     "./save.fcidump", self.int1e, self.int2e, self.norb, self.nelec
        # )
        # fcidump_obj = self.dmrg2.read_fcidump("./save.fcidump")
        # self.mpo = self.dmrg2.get_conventional_qc_mpo(fcidump_obj)
        h1e = np.asarray(self.int1e, dtype=np.float64, order="C")
        g2e = np.asarray(self.int2e, dtype=np.float64, order="C")
        # self.mps.canonicalize()
        # print(self.mps.center)
        # print(self.mps.canonical_form)
        self.dmrg2.initialize_system(n_sites=self.norb, n_elec=self.nelec)
        self.mpo = self.dmrg2.get_qc_mpo(h1e=h1e, g2e=g2e)
        return

    def run_dmrg2(self):
        # self.mps = self.dmrg2.get_random_mps(tag="TEST", bond_dim=10, nroots=1)

        self.energy = self.dmrg2.dmrg(self.mpo, self.mps)
        self.mps_save = self.mps
        return

    def run(self, do_dmrg=True, nuc=0.):
        if do_dmrg:
            self.get_mpo2()
            print("Running DMRG2...")
            self.run_dmrg2()
            mps = self.mps_save
            print(f"DMRG2 energy: {self.energy+nuc}")
        else:
            self.to_fci()
            self._get_fcivec_dict()
            mps = self.scivec2mps()
            print("DMRG not run, energy not calculated.")
        return mps
