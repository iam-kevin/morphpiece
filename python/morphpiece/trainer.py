from unicodedata import normalize as str_normalize
from typing import Iterable, List, Dict, Tuple
import numpy as np
import re

from random import shuffle
from itertools import chain

from tqdm import tqdm

import logging

from .data import TokenTransformer, WordList
from .processor import AlphabetProcessor
from .config import MPConfig

from .core import MorphPiece

ACCEPTED_INIT_METHODS = ['orphan']

logger = logging.getLogger('morphpiece')
logging.basicConfig(level=logging.INFO, 
                    format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
                    datefmt='%H:%M:%S')

DEFAULT = 'default'

def unicode_normalize(text: str, technique: str = 'NFKC'):
    return str_normalize(technique, text)

class MorphPieceTrainer:
    
    def __init__(self,
                 alphabet_processor: AlphabetProcessor,
                 config = DEFAULT,
                 tokenTransformer: TokenTransformer = None):
        if config == DEFAULT:
            self._c = MPConfig()
            
        self.alpp = alphabet_processor
        self._tktf = tokenTransformer
        self._is_trained = False
        self._wordlist = None
        self._morphlist = None
    
    def initalize_matrix(self, shape: Tuple, method='orphan', value=None):
        """Intializing the words that are used in collecting the values        
        """
        
        if method is not None:            
            assert method not in ACCEPTED_INIT_METHODS, \
                "Unsupported method '%s'. Accepted methods are '%r'" % (method, ACCEPTED_INIT_METHODS)
            
            if method == 'orphan':
                value = self._c.cost_orphans
                
        assert isinstance(shape, tuple), "shape must be a tuple"
        assert len(shape) == 2, "shape must only have 2 values"
        
        return np.full(shape, value, dtype=np.float32)
    
    @property
    def is_trained(self):
        return self._is_trained
    
    def _build_cost_matrices(self, a: str, b: str):
        """Building the cost matrices that are going to be used to evaluate 
        the best alignments
        
        Args:
            a: the shorter word to align with `b`
            b: the longer word needed for alignment
        
        Returns:
            - A (N, len(a), len(b)) `np.ndarray` that contains the cost matrixes representing 
            the value of different combination of the words
        
        """
        
        _LEN_A = len(a)
        _LEN_B = len(b)
        
        # indices for strings
        
        offset = 0
        margin = _LEN_B - _LEN_A

        cost_matrix = []
        
        while offset <= margin:
            i = 0 # index for `a`
            # you might want to use a sparse matrix
            cost_indi_matrix = self.initalize_matrix((_LEN_A, _LEN_B), method=None, value=np.inf)
            
            # for each offset, compute the closeness of characters            
            while i < _LEN_A:
                # scan over to the last characters (in iterative method)
                j = i + offset
#                 print(a[i], end=": ")

                while j < _LEN_B:
#                     print("\t%s --> %s" % ((i, j), b[j]))                    
                    if a[i] == b[j]:
                        cost_indi_matrix[i, j] = self._c.cost_twins
                    else:
                        cost_indi_matrix[i, j] = self._c.cost_siblings
                        
                    if j - i >= margin:
                        break
                    j += 1
                i += 1
            
#             print('-'*30)
            # Update scan
            offset += 1
            cost_matrix.append(cost_indi_matrix)
            
        # returns the cost matrix
        return np.array(cost_matrix)
    
    def ideal_cost_matrix(self, cost_matrix: np.ndarray):
        # "summarized" the cost matrix different versions
        # TODO: This will have to change when introducing the sliding property (padded_width = 5)
        cost_matrix = np.min(cost_matrix, axis=0)
        
        return cost_matrix

    def _get_composition(self, cost_matrix: np.ndarray):
        """
        Building the vectors that are used in alignment of the sequences
        """
        # "summarized" the cost matrix different versions
        # TODO: This will have to change when introducing the sliding property (padded_width = 5)
        cost_matrix = self.ideal_cost_matrix(cost_matrix)
                          
        _LEN_A, _LEN_B = cost_matrix.shape       
        assert len(cost_matrix.shape) == 2, "The shape of the cost matrix must be (N, LEN_A, LEN_B)"
    
        
        # building proper alignments
        # ---------------------------
        j = 0 # index for b
        bix = 0 # Index for a
        nix = 0
        
        # to enforce do...while logic
        do_first = True
        
        # state of the values in `b`
        state = {}
        p_from = [] # using this since order is enforced
        p_to = []
        
        while do_first or (bix < _LEN_A and j < _LEN_B):
            
            # for the next set of items, check for the next position
            bix = np.argmin(cost_matrix[bix:, j]) + bix
            
            # There can only be one pair
#             print(state, dict(zip(p_from, p_to)))
#             print('BIX:', bix)
            
            # the remaining (orphan)
            if bix in p_to:
#                 print('YES')
                # build the cost
                if cost_matrix[bix, j] == self._c.cost_siblings:
                    state[j] = self._c.ORPHANS
                    
                elif cost_matrix[bix, j] == self._c.cost_twins:
#                     print(cost_matrix[bix, j], "bix:", bix, "j:", j)
                    # remove previous entry

                    for px in [j-1, j]:
                        if px in p_from:
                            
                            pji = p_from.index(px)

                            # remove entries
                            del p_from[pji]
                            del p_to[pji]
                    
                    # label as orphan
                    state[j-1] = self._c.ORPHANS
                    state[j] = self._c.TWINS
                else:
                    state[j] = self._c.ORPHANS
                    
            else:                
                # build the cost
                if cost_matrix[bix, j] == self._c.cost_twins:
                    state[j] = self._c.TWINS
                elif cost_matrix[bix, j] == self._c.cost_siblings:
                    state[j] = self._c.SIBLINGS
                else:
                    state[j] = self._c.ORPHANS
            
            if state[j] != self._c.ORPHANS:
#                 print('State:', state[j], 'bix:', bix, 'j:', j)
                p_from.append(j)
                p_to.append(bix)
            
            if state[j] == self._c.TWINS:
                bix += 1            
            
            j += 1
            
            if do_first: do_first = False
                
        
        if j < _LEN_B:
            # add the remaining
            for oj in range(j, _LEN_B):
                state[oj] = self._c.ORPHANS
        
        return state, (p_from, p_to)
    
    def _get_cost_from_matrix(self, cost_matrix: np.ndarray):
        """
        Builds the vectors that are used for alignments
        
        This can be used to build the FSM / Knowledge graph
        
        DEV NOTE: THIS IS A POOR LUCKY IMPLEMENTATION
        """
        cost_matrix = self.ideal_cost_matrix(cost_matrix)
        _LEN_B = cost_matrix.shape[1]
        
        assert len(cost_matrix.shape) == 2, "The shape of the cost matrix must be (LEN_A, LEN_B)"
        
        alignment_vector = np.argmin(cost_matrix, axis=1).tolist()
        align_scores = np.min(cost_matrix, axis=1).tolist()

        orphan_ixs = [i for i in range(_LEN_B) if i not in alignment_vector]        
        cost = sum(align_scores) + (self._c.cost_orphans * len(orphan_ixs))

        return cost
    
    def _get_cost(self, a: Iterable, b: Iterable):
        """Estimated rough cost based on closeness of `a` and `b`"""
        if len(a) > len(b):
            # reverse the order making sure 
            #  that `a` is smaller than `b`
            return self._get_cost(b, a)
        
        cost_matrix = self._build_cost_matrices(a, b)
        
        return self._get_cost_from_matrix(cost_matrix)
        
    def _build_alignment(self, a: List[int], b: List[int]):
        if len(a) > len(b):
            # reverse the order making sure 
            #  that `a` is smaller than `b`
            return self._build_alignment(b, a)
        
        cost_matrix = self._build_cost_matrices(a, b)
        
        _shape = cost_matrix.shape[1:]

#         print(self.ideal_cost_matrix(cost_matrix))
        state, pairs = self._get_composition(cost_matrix)

        return state, dict(zip(*pairs))    
        
    def _build_morphlist(self, morphs: List[str]):
        # Sort out the morpheme list by size
        self._morphlist = WordList.from_list(morphs)
        
    def _extract_morphs(self, a: Iterable, b: Iterable, state: Dict[int, str], pairs: Dict[int, int]):
        """Extracting the morphemes from the state and pairs"""
        
        seq = 0
        window_breached = False
        outputs = {}
        v = 0
        
        # if the words dont match up properly, just dont match up at all
        try:
            for i in range(len(b)):
                # Check the matching lengths
                if not window_breached:
                    if state[i] == self._c.TWINS:

                        if i != 0:
                            if state[i - 1] != self._c.TWINS:
                                seq = 0

                        # check the sequence of characters
                        seq += 1

                    window_breached = seq == self._c.min_match_sequence

                # EXTRACTING THE DATA
                if state[i] == self._c.TWINS:

                    if i != 0:
                        if state[i - 1] != self._c.TWINS:
                            v += 1

                    if v not in outputs:
                        outputs[v] = []

                    outputs[v].append(b[i])

                else:
                    if i != 0:
                        if state[i - 1] == self._c.TWINS:
                            v += 1

                    if v not in outputs:
                        # b, a
                        outputs[v] = dict(b=[], a=[])

                    if state[i] == self._c.SIBLINGS:
                        # add the values
                        outputs[v]['a'].append(a[pairs[i]])

                    # SIBLINGS and ORPHANS added value
                    outputs[v]['b'].append(b[i])
        except:
            print(state, pairs)
            assert False, "Something is wrong with this"
                    
        return window_breached, outputs# ... morphemes
    
    def _add_morphs(self, morphs: List[str]):
        """Adding morphemes"""
        for morph in morphs:
            self._morphlist.add(morph)
            
    def clean(self, text: str):
        return text
            
    def _extract_tokens(self, parse_list: List[str]):
        
        # to start with 
        _start_len = len(parse_list)
        pref_token_id = self.alpp[self.alpp.pref_token]
        
        # morphemes
        ffl = WordList([]) # for the proceeding words
        itt = WordList([]) # for the starting words
        
        for i in range(0, _start_len - 1, 2):
            a, b = self.clean(parse_list[i]), self.clean(parse_list[i + 1])
            
            # ensure the shorted text is `a`
            if len(a) > len(b):
                a, b = b, a
                
#             print(a, b)
            
            vals = self.alpp.encode(a), self.alpp.encode(b)
            # output of the text
#             cost = self._get_cost(*vals)
            state, pairs = self._build_alignment(*vals)

            # TODO: this should change when the extraction algorthm 
            #  is changed, when looking at the example of
            #  endapo, anaenda
            #  --> ensure the output is the same
#             vv = self._extract_morphs(a, b, state, pairs)
            sp, dloop = self._extract_morphs(*vals, state, pairs)
            
            # to figure out the word pieces that make up the word
            _a, _b = {}, {}
            ai, bi = 0, 0
            
            for ix in sorted(list(dloop.keys())):
                
                if bi not in _b:
                    _b[bi] = ""
                if ai not in _a:
                    _a[ai] = ""
                    
                if isinstance(dloop[ix], list):
                    uot = dloop[ix]
                    
                    if ix == 0:
                        uot = [pref_token_id] + uot
                        
                    # convert to string
                    b = self.alpp.decode(uot)
                    
                    if len(b) < self._c.min_match_sequence:
                        _b[bi] += b
                        _a[ai] += b
                    else:
                        bi += 1
                        ai += 1
                    
#                     print(b)
                else:
                    uotb, uota = dloop[ix]['b'], dloop[ix]['a']
                    
                    if ix == 0:
                        uotb = [pref_token_id] + uotb
                        uota = [pref_token_id] + uota
                        
                    # convert to string
                    b = self.alpp.decode(uotb)
                    a = self.alpp.decode(uota)                    
                    
                    # add the items to the list
                    if min(len(a), len(b)) < self._c.min_match_sequence:
                        _b[bi] += b
                        _a[ai] += a
                    else:
                        bi += 1
                        ai += 1
                        # add the morphemes
                        if ix == 0:
                            itt.add(a[1:])
                        else:
                            ffl.add(a)
                            
#                     print( "A:", a, "B:", b)
                    
                if len(b) >= self._c.min_match_sequence:
                    if ix == 0:
                        itt.add(b[1:])
                    else:
                        ffl.add(b)
                        
            
            # add values:
            for tx in chain.from_iterable((_a.values(), _b.values())):
                if tx.find(self.alpp.pref_token) == 0:
                    # start of the text
                    itt.add(tx[1:])
                else:
                    ffl.add(tx)
                       
        # returns the output with the former text and 
        #  present text
        return itt, ffl
    
    def _extract_tokens_early(self, parse_list: List[str]):
        """Extract the tokens based on cost computations
        
        This should be a greedy early computation 
        that tries to obtain the new values 
        """
        # to start with 
        _start_len = len(parse_list)
        pref_token_id = self.alpp[self.alpp.pref_token]
        
        # morphemes
        ffl = WordList([]) # for the proceeding words
        itt = WordList([]) # for the starting words
        
#         for i 
        
        return itt, ffl
    
    def train(self, wordlist: WordList, morphs: List[str] = None) -> MorphPiece:
        # building the wordlist
        self._wordlist = wordlist
        
        assert not self.is_trained, "The Trainer has already ran! You are unable to run this again."
        
        # 1. MASK THE PRE-SET MORPHEMES IN THE LIST OF WORDS
        # Build the morphemes list
        if morphs is None:
            morphs = []
        
        logger.info('Initializing a morphemes list')
        self._build_morphlist(morphs)
        
        _parse_list = None
        
        # get the list to parse through
        if self._c.alphabetized:
            _parse_list = self._wordlist.get_alphabetized_list()
        else:
            _parse_list = self._wordlist.get_words()
        
        # traversing through the list to mask the send morphemes
        if len(self._morphlist) > 0:
            # update the list by masking the words
            logger.info('Masking pre-defined morphemes')
            
            for morph in self._morphlist.get_longest_first_list():
                for ix, word in enumerate(_parse_list):
                    masked = self.alpp.mask_characters(word, morph, coded=False)
                    
                    # update the parse list with the added morpheme
                    if masked != word:
                        _parse_list[ix] = masked
                        self._morphlist.add(morph)
#         print(_parse_list)
        
        # 2. ALIGN ALL THE WORDS IN THE PAIRS
        logger.info('Aligning all the words, pair-by-pair, in the list and building morphemes list')
        
#         ab, bc =  self._extract_tokens(_parse_list)
        pref_vals, base_suf_vals =  self._extract_tokens(_parse_list)
        
        init_filter = pref_vals.get_alphabetized_list(True, False)
        base_filter = base_suf_vals.get_alphabetized_list(True, False)
            
        assert self._c.iters >= 1, "Configuration for `iters` must be greater than or equal to 1"
        
        logger.info('Passing alignments iteratively to build morpheme list [Iterations: %s]' % (self._c.iters))
        
        # iterating
        for i in tqdm(range(self._c.iters)):
            _bb, _cc = self._extract_tokens(init_filter)
            bb, cc = self._extract_tokens(base_filter)
            
            init_filter = _bb.get_alphabetized_list(True, False)
            
            base_filter = _cc.get_alphabetized_list(True, False)  + \
                            cc.get_alphabetized_list(True, False) + \
                                bb.get_alphabetized_list(True, False)
            
            # add the new values
            
#             print(init_filter, base_filter)
            
            # adding the values
            pref_vals.add_list(init_filter, skip_count=True)
            base_suf_vals.add_list(base_filter, skip_count=True)
            
#             print("-" * 50)
            
        # set the training value to true
        self._is_trained = True
        
        logger.info('Training complete')
        logger.info('Morphemes summary: \n[Prefix morphemes: %d] \n[Base morphemes: %d]' % (len(pref_vals), len(base_suf_vals)))
        
        # build the morphpiece objects
        self.build(pref_vals, base_suf_vals)
        
        return self.get()
        
    def build(self, prefix_morphemes: WordList, base_morphemes: WordList):
        self.mp = MorphPiece(prefix_morphemes, base_morphemes, self.alpp)
        
    def get(self):
        """Gets the morpheme piece object"""
        assert self.is_trained, "The %s is not trained" % (type(self).__name__)
        
        return self.mp

