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
    def random(num_hdvs, hdv_len):
        pass

    @staticmethod
    @abstractmethod
    def zero(num_hdvs, hdv_len):
        pass


class BinaryHDV(AbstractHDV):
    def __init__(self, hdv_len, data):
        super().__init__(hdv_len, data, HDVClass.BINARY)

    def bind(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class bind mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return BinaryHDV(self.hdv_len, self.data & other.data)

    def bundle(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class bundle mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return BinaryHDV(self.hdv_len, self.data ^ other.data)

    def inv_bind(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class inv_bind mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return BinaryHDV(self.hdv_len, self.data & (~other.data))

    # Only for binary field $$(F_2^n)$$
    # XOR is involutive
    def inv_bundle(self, other):
        return bundle(self, other)

    def similarity(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class similarity mismatch"
        assert other.hdv_len == self.hdv_len, "operands must match length"
        return jax.lax.reduce_sum(self.data & other.data, [0])


    def norm(self):
        return self.similarity(self)


    def random(num_hdvs, hdv_len, key=jrnd.key(72)):
        vecs = jrnd.bernoulli(key, shape=(num_hdvs, hdv_len)).astype(jnp.uint16)
        return [BinaryHDV(hdv_len, v) for v in vecs]

    def zero(num_hdvs, hdv_len):
        return [BinaryHDV(hdv_len, jnp.zeros(hdv_len).astype(jnp.uint16)) for _ in range(num_hdvs)]

    def hamming_similarity(self, other):
        assert other.hdv_class == HDVClass.BINARY, "class similarity mismatch"
        assert other.hdv_len == self.hdv_len, "operand length mismatch"
        return other.hdv_len - jax.lax.reduce_sum(self.data ^ other.data, [0])



class ComplexHDV(AbstractHDV):
    def __init__(self, hdv_len, data):
        super().__init__(hdv_len, data, HDVClass.COMPLEX)

    def bind(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class bind mismatch"
        assert other.hdv_len == self.hdv_len, "operand length mismatch"
        return ComplexHDV(self.hdv_len, self.data * other.data)

    def bundle(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class bundle mismatch"
        assert other.hdv_len == self.hdv_len, "operand length mismatch"
        return ComplexHDV(self.hdv_len, self.data + other.data)

    # division; self/other
    def inv_bind(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class inv_bind mismatch"
        assert other.hdv_len == self.hdv_len, "operand length mismatch"
        return ComplexHDV(self.hdv_len, self.data / other.data)

    # Returns self - other
    def inv_bundle(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class inv_bind mismatch"
        assert other.hdv_len == self.hdv_len, "operand length mismatch"
        return ComplexHDV(self.hdv_len, self.data - other.data)


    # Complex inner product
    # <self, other.conjugate()> = self.T * other.conjugate()
    def similarity(self, other):
        assert other.hdv_class == HDVClass.COMPLEX, "class similarity mismatch"
        assert other.hdv_len == self.hdv_len, "operand length mismatch"
        return jnp.inner(self.data, other.data.conj())


    def norm(self):
        return self.similarity(self)


    def random(num_hdvs, hdv_len, key=jrnd.key(72)):
        vecs = jrnd.uniform(key, shape=(num_hdvs, hdv_len), maxval=2*jnp.pi)
        vecs = jnp.exp(vecs*1j)
        return [ComplexHDV(hdv_len, v) for v in vecs]


    def zero(num_hdvs, hdv_len):
        return [ComplexHDV(hdv_len, jnp.ones(hdv_len).astype(jnp.complex64)) for _ in range(num_hdvs)]


