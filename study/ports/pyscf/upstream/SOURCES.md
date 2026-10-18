# The original: PySCF 2.5.0

- Project: PySCF, http://github.com/pyscf/pyscf (http://www.pyscf.org).
- Version: 2.5.0, installed with pip (wheel) into Python 3.11.
- License: Apache-2.0. `LICENSE` and `NOTICE` here are the copies from the
  installed `pyscf-2.5.0.dist-info`.
- Libraries it runs on this machine: NumPy 1.24.4 and SciPy 1.12.0 against
  OpenBLAS (ILP64 build, `openblas64_`); h5py 3.9.0 for the checkpoint file.
  The wheel's C libraries are built without OpenMP (PySCF warns "OpenMP is not
  available" when `lib.num_threads(1)` is called), so its C code runs on one
  thread whatever the setting.
- Nothing of PySCF is copied into the port. The Viv side is lib/hf, written
  independently; this port runs PySCF as the oracle and as the speed
  reference.

The files on the ported path, as installed (sha256):

    f596627c10f1fc46197f8a82ac98bdf5980ed5315c65ae2ee5546e6a3242904e  pyscf/scf/hf.py
    0505f9a8ff1b57dce3c1cf8fa6a23a8f2172652a7b39cd262a287ede24899772  pyscf/scf/uhf.py
    72806b396f10105a62457d2f49ed266e2043c28086b1e257fbb2aacfe48a1bb7  pyscf/scf/diis.py
    25a297413195e5917ce934f4326abddc99fbf8e024bc5237947918f3e4fae0f2  pyscf/scf/_vhf.py
    f12173266050bd1746e22f41f086415fc42c5fc0c87cd69a5063acdb5f7a3bd5  pyscf/lib/diis.py
    ea96b67c38adaea298cc5e424fa4ab0fe86a1d986bf7c5f9c2f5597656a34e65  pyscf/gto/moleintor.py
    13dc602412b8431f12780760c3ae01cab116cf59b993ba33a4cc090e733e59b6  pyscf/gto/mole.py
    63fb015fdcf401482e34ceb11dd9eda3abfcba608112c8bbecac1461d2adb520  pyscf/lib/libcvhf.dylib
    bb154f64a081c85d74b117102f888700ad0a6314af98a47aab493d290d51d4b9  pyscf/lib/libcgto.dylib
    916987037891669b05ea7b364636095003d82c006754747a68c5619f3c009c55  pyscf/lib/deps/lib/libcint.6.dylib

libcint's source is not on this machine (the wheel carries the library
only), so its lines are not counted (results/lines.json says so).
