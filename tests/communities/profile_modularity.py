import cProfile

import numpy as np
import xgi

p = np.array([[0.3, 0.01], [0.01, 0.3]])
H = xgi.uniform_HSBM(30, 2, p, [15, 15])

# clusters = xgi.simple_cnm(H)

cProfile.run("xgi.simple_cnm(H)", "cnm.profile")
