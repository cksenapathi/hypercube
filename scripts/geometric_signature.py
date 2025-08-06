import jax
from itertools import combinations
import jax_hdv


# Assuming a time-series is a function/geometric rough path with
#   bounded variation, convert the time-series into a signature 
# Signature is a element of a graded tensor algebra
#   Encodes graded variation (graded differential forms)

# time series is a list of HDV objects
def timeseries_to_signature(timeseries):
    for l in range(2, len(timeseries)+1):
        yield list(combinations(timeseries, l))

sequence = list(range(7))

value = [s ** 2 for s in sequence]

print(value)

# Order 1
dx_t = [value[i+1] - value[i] for i in range(len(value)-1)]

print(dx_t)

order1 = sum(dx_t)

print(order1)

# Order 2
running_dx_t = [sum(dx_t[:i+1]) for i in range(len(dx_t))]
print(len(dx_t), len(running_dx_t))

print(dx_t, running_dx_t)

print(sum([0]))
# something doesn't feel right about the signature
# there shouldn't be a way to generate infinite information out of finite information
# extract as much information as possible
# but infinite information doesn't feel right
# 






