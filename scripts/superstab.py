# Superstabilizing network for fault tolerance

# Model as a Simplex must run async/event based wrt interface
model_simplex = None # 0 HDV/Empty Simplex
neighbors = []

while True:
    # poll sensors for value
    obs = get_obs()

    # Sure, get them into a standard format
    # The encoding will also need to be updated
    obs_hdv = encode_obs(obs, encoding_hdvs)

    # drop into the map
    # local to previous 0-simplex
    # bottom up dp approach w signature
    # 
    model_simplex.add(obs_hdv)




