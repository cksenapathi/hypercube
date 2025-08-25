import jax.numpy as jnp

simplex = {}

# A k-simplex has k+1 elements each of the lower grade
# Could theoretically be sub-simplex of any number of (k+1)-simplices
# Each simplex holds pointers to neighbors of 1 grade down and 1 grade up
# All computation must only operate on local (data, neighbors, action)
class Simplex:
    def __init__(self, grade, down_grade_nbrs=[], up_grade_nbrs=[], data=None):
        self.grade = grade
        self.down_grade_nbrs = down_grade_nbrs # ordered
        self.up_grade_nbrs = up_grade_nbrs
        self.data = data



# each new point must add 0-sim
for i in range(5):
    simplex.get(0, []).append(i)

    






