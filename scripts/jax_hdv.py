import jax
import jax.random as jrnd
import jax.numpy as jnp
from enum import IntEnum
from abc import ABC

class HDVClass(IntEnum):
    BINARY = 1
    BIPOLAR = 2
    COMPLEX = 3


key = jrnd.key(1534)


def random(n_vecs, hdv_len, hdv_class):
    if hdv_class == HDVClass.BINARY:
        return jrnd.bernoulli(key, shape=(n_vecs, hdv_len))
    elif hdv_class == HDVClass.BIPOLAR:
        return 1 - 2*jnp.astype(jrnd.bernoulli(key, shape=(n_vecs, hdv_len)))
    elif hdv_class == HDVClass.COMPLEX:
        return jax.lax.exp(2*jnp.pi*jrnd.uniform(key, shape=(n_vecs, hdv_len)))
    else:
        print("ERROR: UKNOWN HDV_CLASS TYPE '{}'. MUST BE ONE OF {}".format(hdv_class, list(HDVClass)))
        return None


random(1, 15, "HOLOGRAPHIC")
print(random(1, 15, HDVClass.COMPLEX))


