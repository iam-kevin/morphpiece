class MPConfig:
    """Holds the values for determining the
    configuration of the values"""
    cost_twins = 0
    cost_siblings = 1.5
    cost_orphans = 1
    
    TWINS = 'TWINS'
    SIBLINGS = 'SIBLINGS'
    ORPHANS = 'ORPHANS'
    
    # ACTUAL HYPERPARAMETERS
    min_match_sequence = 3
    min_seq_length = 5
    min_stem_length = 4
    
    # Sort the corpus in aphabetical order
    alphabetized = False
    
    
    iters = 4 # iterations to build the vocabulary