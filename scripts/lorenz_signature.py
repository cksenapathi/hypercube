import jax.random as jrnd
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
from jax import jit

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from jax_hdv import BinaryHDV
# how is increasingly strict algebraic structure
# attached to geometric differentials

# Converts sequence of f32's to sequence of 32-bit binary strings
def f32_to_binary(num_array):
    bitstrings = []
    for num in num_array:
        byte_string = struct.pack("@f", num)
        bitstring = "".join([format(i, "08b") for i in byte_string])
        bitstrings.append(bitstring)


@jit 
def lorenz(t, X, args):

    x, y, z = X

    xdot = 10 * (y - x)
    ydot = x * (28 - z) - y
    zdot = x * y - (8./3.) * z

    return jnp.array([xdot, ydot, zdot])

# Generate time trajectory for testing
term = ODETerm(lorenz)
solver = Dopri5()
X0 = jnp.array([1, 1, 1])
saveat = SaveAt(ts=jnp.linspace(0,50,2500))
solution = diffeqsolve(term, solver, t0=0, t1=50, dt0=0.05,y0=X0, saveat=saveat)

#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(*solution.ys.T)
#plt.show()

data = jnp.concatenate([jnp.expand_dims(solution.ts, -1), solution.ys], axis=-1)
#t = solution.ts
#ys = solution.ys



#print(f"t:{t.shape}, {t.dtype}; ys:{ys.shape}, {ys.dtype}")

for i, y in enumerate(data):
    f32_to_binary(y)
    if i == 3: break


