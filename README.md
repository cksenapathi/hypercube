# hypercube platform for tesseract
- progress:
  - binary and complex hdvs; need real (representing topological vector space)
  - implementing simplicial complex as container for CHS
  - composition of ASC must match composition of PHS

- todo:
  - dynamic approach to signature construction
    - each new blanket state must provide information on sequence
    - bottom up approach to higher order effects --> dynamic stopping
  - random sampling in hdv classes needs to cycle prng key

- goals:
  - model, data, and measure compositionality
  - continual local learning by message passing formulation

- notes: 
  - jaxga as jax backend for geometric algebra
  - liboqs-python for pqc in python -- build c lib w/ python bindings
  - tree-math -- jax add-on for ops over pytree objects

- implementation:
  - input data encoded by hdvs
  - non-param belief prop (npbp)
    - function vecs are mapped to local action space
    - function vecs are probability distribution
    - local action space is given by linear matrix in standard npbp
    - for more complex actions, the question is how does one probability distribution (wavefunction/state) transform into another
  - conformal/port hamiltonian system (CHS/PHS)
    - a general dynamical description of flow of action
    - each local action represents a hierarchical CHS that changes the fn vec
    - each k-simplex holds a local state which describes an action in conjuction with the action 
    - 0-simplex: boundary data; 1-simplex: differential
    - higher order simplex for higher order differentials for higher order interactions
    - each conformal hamiltonian is a density process generator
    - the world is dual to the agent, so the world provides action, and the density must learn how to mirror
    - therefore, k-simplex encodes a 2\*k - dimensional CHS; 1-dim is constant function i.e., data generator 
  - what's learned
    - which states, are associated with with processes
    - this must be a process algebra

