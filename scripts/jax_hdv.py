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


class AbstractHDV(ABC):
    @abstractmethod
    def __init__(self, hdv_len, data, hdv_class):
        self.hdv_len = hdv_len
        self.data = data
        self.hdv_class = hdv_class

    @abstractmethod
    def bind(self, other):
        pass

    @abstractmethod
    def bundle(self, other):
        pass

    @abstractmethod
    def inv_bind(self, other):
        pass

    @abstractmethod
    def inv_bundle(self, other):
        pass
    
    @abstractmethod
    def similarity(self, other):
        pass

    @abstractmethod
    def norm(self, other):
        pass


    @staticmethod
    @abstractmethod
    def random(self, num_hdvs, hdv_len):
        pass


class BinaryHDV(AbstractHDV):
    def __init__(self, hdv_len, data):
        super().__init__(hdv_len, data, HDVClass.BINARY)

    def bind(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class bind mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return BinaryHDV(self.hdv_len, self.data & other.data, HDVClass.BINARY)

    def bundle(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class bundle mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return BinaryHDV(self.hdv_len, self.data ^ other.data, HDVClass.BINARY)

    def inv_bind(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class inv_bind mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return BinaryHDV(self.hdv_len, self.data & (~other.data), HDVClass.BINARY)

    # Only for binary field $$(F_2^n)$$
    # XOR is involutive
    def inv_bundle(self, other):
        return bundle(self, other)

    def similarity(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class similarity mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return self.hdv_len - jax.lax.reduce_sum(self.data ^ other.data, [0])


    def norm(self):
        return similarity(self)


    def random(self, num_hdvs, hdv_len, hdv_class=HDVClass.BINARY, key=jrnd.key(72)):
        vecs = jrnd.bernoulli(key, shape=(num_hdvs, hdv_len))
        return [BinaryHDV(hdv_len, v, HDVClass.BINARY) for v in vecs]


class ComplexHDV(AbstractHDV):
    def __init__(self, hdv_len, data):
        super().__init__(hdv_len, data, HDVClass.COMPLEX)

    def bind(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class bind mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return ComplexHDV(self.hdv_len, self.data & other.data, HDVClass.COMPLEX)

    def bundle(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class bundle mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return ComplexHDV(self.hdv_len, self.data ^ other.data, HDVClass.COMPLEX)

    def inv_bind(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class inv_bind mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return ComplexHDV(self.hdv_len, self.data & (~other.data), HDVClass.COMPLEX)

    # Only for binary field $$(F_2^n)$$
    # XOR is involutive
    def inv_bundle(self, other):
        return bundle(self, other)

    def similarity(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class similarity mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return self.hdv_len - jax.lax.reduce_sum(self.data ^ other.data, [0])


    def norm(self):
        return similarity(self)


    def random(self, num_hdvs, hdv_len, hdv_class=HDVClass.COMPLEX, key=jrnd.key(72)):
        vecs = jrnd.bernoulli(key, shape=(num_hdvs, hdv_len))
        return [ComplexHDV(hdv_len, v, HDVClass.COMPLEX) for v in vecs]

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

