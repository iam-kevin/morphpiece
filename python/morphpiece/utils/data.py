from typing import Tuple, List, Iterable, Any

def generate_negative_samples(samples: int):
    pass

def generate_context_samples(tokens_seq: List[str], window_size, indexed=False):
    """Genarate context tokens for the corresponding token in a sequence"""
    _ls = None
    if indexed:
        _ls = range(len(tokens_seq))
    
    for ix, word in enumerate(tokens_seq):
        # get left context
        left_indices = max(0, ix - window_size)
        right_indices = min(ix + window_size + 1, len(tokens_seq) + 1)
        
        if indexed:
            yield ix, list(_ls[left_indices:ix]) + list(_ls[ix + 1:right_indices])
        else:
            yield word, tokens_seq[left_indices:ix] + tokens_seq[ix + 1:right_indices]