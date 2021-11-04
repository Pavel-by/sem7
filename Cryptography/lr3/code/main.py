import numpy as np
from tabulate import tabulate

print({_:lambda s: int(s, 16) for _ in range(16)})

S_BOX = np.loadtxt('tables/s_box', delimiter=' ', dtype=np.int32, converters={_:lambda s: int(s, 16) for _ in range(16)}, )
S_BOX_INV = np.loadtxt('tables/s_box_inv', delimiter=' ', dtype=np.int32, converters={_:lambda s: int(s, 16) for _ in range(16)}, )



print(S_BOX)
