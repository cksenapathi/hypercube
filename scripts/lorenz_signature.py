import jax.random as jrnd
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Dopri5, SaveAt
from jax import jit
import struct

import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

from jax_hdv import BinaryHDV, HDVClass
# how is increasingly strict algebraic structure
# attached to geometric differentials

# Converts sequence of f32's to sequence of 32-bit binary strings
def f32_to_bitstring(num_array):
    bitstrings = []
    
    for i, num in enumerate(num_array):
        #idx_bits = "{:02b}".format(i)
        byte_string = struct.pack("@f", num)
        bitstring = "".join([format(i, "08b") for i in byte_string])
        bitstrings.append(bitstring)
    
    #print(bitstrings)
    return bitstrings

def bitstring_to_f32(bitstring):
    byte_repr = int(bitstring, 2).to_bytes(len(bitstring)//8)
    
    return struct.unpack("@f", byte_repr)


# List of bitstring repr of data and encoding BinaryHDVs
# Each bitstring is encoded as an f32 with 1 hdv for each bit
#   bundled dependent on bit value
# Encoded bitstring is then bound with index hdv
#   ensures same bit in different f32's don't overwrite as xor involutes
# All bitstring hdvs bundled into 1 hdv as set repr
# Can be thought of as hashmap or list where idx (key) returns corr. float (value)
def bitstring_to_hdv(bitstrings, encoding_hdvs, idx_hdvs):
    hdv_len = encoding_hdvs[0].hdv_len
    vec = BinaryHDV.zero(1, hdv_len)[0]

    for j, bitstr in enumerate(bitstrings):
        
        idx_hdv = idx_hdvs[j]
        bitstr_hdv = BinaryHDV.zero(1, hdv_len)[0]

        for i, b in enumerate(bitstr):
            if b == '1': bitstr_hdv = bitstr_hdv.bundle(encoding_hdvs[i])
    
        bitstr_hdv = bitstr_hdv.bind(idx_hdv)

        vec = vec.bundle(bitstr_hdv)

    return vec


def hdv_to_bitstring(hdv, encoding_hdvs, idx_hdvs):
    pass


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

#t = solution.ts
#ys = solution.ys

#ax = plt.figure().add_subplot(projection='3d')
#ax.plot(*ys.T)
#plt.show()

data = jnp.concatenate([jnp.expand_dims(solution.ts, -1), solution.ys], axis=-1)

data_len = 4
data_prec = 32
hdv_len = 10000

idx_hdvs = BinaryHDV.random(data_len, hdv_len)
encoding_hdvs = BinaryHDV.random(data_prec, hdv_len)

for i, y in enumerate(data):
    data_bitstrings = f32_to_binary(y)
    encoded_hdv = bitstring_to_hdv(data_bitstrings, encoding_hdvs, idx_hdvs)
    print(encoded_hdv.data)
    if i == 3: break

