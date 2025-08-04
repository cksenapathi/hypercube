import jax
import jax.random as jrnd
import jax.numpy as jnp
from enum import IntEnum
from dataclasses import dataclass
from abc import ABC, abstractmethod


class HDVClass(IntEnum):
    BINARY = 1
    BIPOLAR = 2
    COMPLEX = 3


key = jrnd.key(1534)


@dataclass
class AbstractDataClass(ABC):
    def __new__(cls, *args, **kwargs):
        if cls == AbstractDataClass or cls.__bases__[0] == AbstractDataClass:
            raise TypeError("Cannot instantiate abstract class")
        return super().__new__(cls)

@dataclass
class HDV(AbstractDataClass):
    hdv_len: int
    data: jax.Array
    hdv_class: HDVClass

    @staticmethod
    def random(n_vecs, hdv_len, hdv_class):
        if hdv_class == HDVClass.BINARY:
            vecs = jrnd.bernoulli(key, shape=(n_vecs, hdv_len))
            return [BinaryHDV(hdv_len, v, HDVClass.BINARY) for v in vecs]
        elif hdv_class == HDVClass.BIPOLAR:
            vecs = 1 - 2*jnp.astype(jrnd.bernoulli(key, shape=(n_vecs, hdv_len)), jnp.int8)
            return [BipolarHDV(hdv_len, v, HDVClass.BIPOLAR) for v in vecs]
        elif hdv_class == HDVClass.COMPLEX:
            vecs = jax.lax.exp(2*jnp.pi*jrnd.uniform(key, shape=(n_vecs, hdv_len)))
            return [ComplexHDV(hdv_len, v, HDVClass.COMPLEX) for v in vecs]
        else:
            print("ERROR: UKNOWN HDV_CLASS TYPE '{}'. MUST BE ONE OF {}".format(hdv_class, list(HDVClass)))
            return None

   

@dataclass
class BinaryHDV(HDV):
    pass 
    
@dataclass
class BipolarHDV(HDV):
    pass

@dataclass
class ComplexHDV(HDV):
    pass

#def random(n_vecs, hdv_len, hdv_class):
#    if hdv_class == HDVClass.BINARY:
#        return jrnd.bernoulli(key, shape=(n_vecs, hdv_len))
#    elif hdv_class == HDVClass.BIPOLAR:
#        return 1 - 2*jnp.astype(jrnd.bernoulli(key, shape=(n_vecs, hdv_len)), jnp.int)
#    elif hdv_class == HDVClass.COMPLEX:
#        return jax.lax.exp(2*jnp.pi*jrnd.uniform(key, shape=(n_vecs, hdv_len)))
#    else:
#        print("ERROR: UKNOWN HDV_CLASS TYPE '{}'. MUST BE ONE OF {}".format(hdv_class, list(HDVClass)))
#        return None


# Bundle is roughly a join
# Information from both constructions is preserved
def bundle(hdv_a, hdv_b):
    assert hdv_a.hdv_class == hdv_b.hdv_class, "HDVs must be same hdv_class"
    assert hdv_a.hdv_len == hdv_b.hdv_len, "HDVs must be same length" 

    hdv_class = hdv_a.hdv_class
    hdv_len = hdv_a.hdv_len    

    if hdv_class == HDVClass.BINARY:
        return BinaryHDV(hdv_len, hdv_a.data ^ hdv_b.data, HDVClass.BINARY)  
    elif hdv_class == HDVClass.BIPOLAR:
        return BipolarHDV(hdv_len, hdv_a.data + hdv_b.data, HDVClass.BIPOLAR)
    elif hdv_class == HDVClass.COMPLEX:
        return ComplexHDV(hdv_len, hdv_a.data + hdv_b.data, HDVClass.COMPLEX)
    else:
        print("ERROR: UNKNOWN HDV_CLASS '{}' MUST BE ONE OF {}".format(hdv_class, list(HDVClass)))
        return None

# Bind is roughly a meet
# Dissimilar from both
# Approximates tensor product?? -- idk what connection here is
def bind(hdv_a, hdv_b):
    assert hdv_a.hdv_class == hdv_b.hdv_class, "HDVs must be same hdv_class"
    assert hdv_a.hdv_len == hdv_b.hdv_len, "HDVs must be same length" 

    hdv_class = hdv_a.hdv_class
    hdv_len = hdv_a.hdv_len    

    if hdv_class == HDVClass.BINARY:
        return BinaryHDV(hdv_len, hdv_a.data & hdv_b.data, HDVClass.BINARY)  
    elif hdv_class == HDVClass.BIPOLAR:
        return BipolarHDV(hdv_len, hdv_a.data * hdv_b.data, HDVClass.BIPOLAR)
    elif hdv_class == HDVClass.COMPLEX:
        return ComplexHDV(hdv_len, hdv_a.data * hdv_b.data, HDVClass.COMPLEX)
    else:
        print("ERROR: UNKNOWN HDV_CLASS '{}' MUST BE ONE OF {}".format(hdv_class, list(HDVClass)))
        return None

# similarity measure
# Roughly inner product for Hilbert Space
def similarity(hdv_a, hdv_b):
    assert hdv_a.hdv_class == hdv_b.hdv_class, "HDVs must be same hdv_class"
    assert hdv_a.hdv_len == hdv_b.hdv_len, "HDVs must be same length" 

    hdv_class = hdv_a.hdv_class
    hdv_len = hdv_a.hdv_len    

    if hdv_class == HDVClass.BINARY:
        return jax.lax.reduce_sum(hdv_a.data & hdv_b.data, [0]) 
    elif hdv_class == HDVClass.BIPOLAR:
        return jax.lax.dot(hdv_a.data, hdv_b.data)
    elif hdv_class == HDVClass.COMPLEX:
        return jax.lax.dot(hdv_a.data, hdv_b.data.conj())
    else:
        print("ERROR: UNKNOWN HDV_CLASS '{}' MUST BE ONE OF {}".format(hdv_class, list(HDVClass)))
        return None




#random_binary = HDV.random(3, 15, HDVClass.BINARY)
#random_bipolar = HDV.random(3, 15, HDVClass.BIPOLAR)
#random_complex = HDV.random(3, 15, HDVClass.COMPLEX)
#
#print("Random binary", random_binary)
#print("Random bipolar", random_bipolar)
#print("Random complex", random_complex)
#
#
#hdv_a = random_binary[0]
#hdv_b = random_binary[1]
#hdv_c = random_complex[0]
#
#bundle(hdv_a, hdv_b)
#bundle(hdv_a, hdv_c)
#

