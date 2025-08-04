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

diffs = [value[i+1] - value[i] for i in range(len(value)-1)]

print(diffs)

order1 = sum(diffs)

print(order1)


# something doesn't feel right about the signature
# there shouldn't be a way to generate infinite information out of finite information
# extract as much information as possible
# but infinite information doesn't feel right
# 






