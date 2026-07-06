
def add_state_dicts(state_a: dict, state_b: dict) -> dict:
    """
    Computes the element-wise sum of two model state dictionaries without 
    mutating the original inputs.
    
    Args:
        state_a: An OrderedDict of model parameters.
        state_b: An OrderedDict of model parameters with identical keys and shapes.
        
    Returns:
        A new dictionary containing the summed tensors.
    """
    result_state = {}
    
    # Since we mathematically guarantee both dicts come from the same architecture,
    # we can iterate over the keys of one to access both.
    for key in state_a.keys():
        
        # The standard '+' operator allocates a fresh tensor in memory,
        # ensuring we do not accidentally overwrite the caller's data.
        result_state[key] = state_a[key] + state_b[key]
        
    return result_state

def scale_state_dict(state_dict: dict, weight: float) -> dict:
    """
    Multiplies every tensor in a model state dictionary by a scalar weight,
    returning a new dictionary to preserve the original memory.
    
    Args:
        state_dict: An OrderedDict of model parameters/buffers.
        weight: The scalar float value to multiply against every tensor.
        
    Returns:
        A new dictionary containing the scaled tensors.
    """
    scaled_state = {}
    
    # Iterate through every parameter and buffer in the dictionary
    for key, tensor in state_dict.items():
        
        # The standard '*' operator allocates a new tensor in memory.
        # This keeps the original client state completely untouched.
        scaled_state[key] = tensor * weight
        
    return scaled_state

def aggregate_weighted_average(client_states: list, client_sample_counts: list) -> dict:
    """
    Fuses multiple client models into a single global model via a sample-weighted average.
    
    Args:
        client_states: A list of state dictionaries from locally trained models.
        client_sample_counts: A list of integers representing the number of data 
                              samples each client trained on.
                              
    Returns:
        A new state dictionary representing the updated global model.
    """
    total_samples = sum(client_sample_counts)
    
    global_state = None
    
    # Python's zip allows us to iterate through both lists perfectly in sync
    for state, count in zip(client_states, client_sample_counts):
        
        # Calculate the proportional weight of this client
        weight = count / total_samples
        
        # Scale the client's parameters (allocates fresh memory via your Step 15 function)
        scaled_state = scale_state_dict(state, weight)
        
        # Accumulate the scaled state into the global state
        if global_state is None:
            # First client establishes the baseline dictionary
            global_state = scaled_state
        else:
            # Subsequent clients are added element-wise via your Step 14 function
            global_state = add_state_dicts(global_state, scaled_state)
            
    return global_state