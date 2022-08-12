from pypfm import PFMLoader
import numpy as np
import sys


loader = PFMLoader(color=False, compress=False)
pfm = loader.load_pfm(sys.argv[1])
print(pfm.shape)
print(pfm[123][456])
