from typing import Tuple, List, Iterable, Any, Optional
from random import shuffle

def generate_context_samples(tokens_seq: List[str], 
                             window_size, 
                             indexed=False, 
                             sample_negatives=False, 
                             random: bool=True,
                             top_neg_pick: Optional[int] = 3):
    """Genarate context tokens for the corresponding token in a sequence"""
    _ls = None
    if indexed:
        _ls = range(len(tokens_seq))
    
    for ix, word in enumerate(tokens_seq):
        # get left context
        left_indices = max(0, ix - window_size)
        right_indices = min(ix + window_size + 1, len(tokens_seq) + 1)
        
        output = dict()
        
        neg_samples = None
        if sample_negatives:
            neg_samples = list(range(0,left_indices)) + list(range(right_indices+1,len(tokens_seq)))
            
            if random:
                # shuffles the list
                shuffle(neg_samples)
                
            if top_neg_pick is not None:
                # picks the top `max_random_pick`
                neg_samples = neg_samples[:top_neg_pick]                
                
        if indexed:
            key = ix
            output['context']= list(_ls[left_indices:ix]) + list(_ls[ix + 1:right_indices])
            if sample_negatives: 
                output['neg_samples'] = [_ls[i] for i in neg_samples]
        else:
            key = word
            output['context']= tokens_seq[left_indices:ix] + tokens_seq[ix + 1:right_indices]
            if sample_negatives:
                output['neg_samples'] = [tokens_seq[i] for i in neg_samples]
                          
        
        
        yield key, output