from ..data import _ReferenceVocabulary as ReferenceVocabulary, LowercaseSwahiliReferenceVocabulary
from typing import List, Union

from ..data.reference_vocabulary import UNK_CHAR
from itertools import chain

from ..data import TokenTransformer, BasicTokenTransformer

# This out of scope token is used its unable to map the 
# index back to the number
OOS_TOKEN = '%' # '[OOS]'
MASK_TOKEN = '&' # '[MASK]'
PREF_TOKEN = "_" # this is would usually used to prepend the character representing the first letter of the word

# this are used in marking the start and end of a word
# TODO: Work with this later
TOKEN_START = '<w>'
TOKEN_END = '</w>'

_rfv = {
    'swahili': LowercaseSwahiliReferenceVocabulary
}

class AlphabetProcessor:
    """This component takes the things that make up the text vocabulary to
    build a character mapper that is useful in constructing match the characters in string
    
    With this, we expect to make it possible to convert the words `alienda` if the following
    dictionary of word is built
    
        ```
        { 
            a: 0,
            d: 1
            e: 2,
            i: 3,
            l: 4,
            n: 5
        }
        ```
    
    'alienda' -> [0, 4, 3, 2, 5, 1, 0]
    
    NOTE: the other stuff are not yet implemented (like `base_characters`, ... )
    """
    def __init__(self, 
                 ref_vocab: Union[str, ReferenceVocabulary], 
                 base_characters: List[str]=None,
                 base_numbers: List[str]=None,
                 base_punctuations: List[str]=None,
                 unk_char_token: str = UNK_CHAR,
                 token_transformer: TokenTransformer = BasicTokenTransformer()):
        
        if isinstance(ref_vocab, str):
            if ref_vocab in _rfv:
                ref_vocab = _rfv[ref_vocab]()
            else:
                raise ValueError('Unsupported reference vocab type. Expected one in %r' % (_rfv))
        
        # this code only supports the `LowerCaseSwahiliRV` reference vocab built for
        # the swahili language (this is the only biasing thing)
        assert isinstance(ref_vocab, LowercaseSwahiliReferenceVocabulary), \
            "Expected `ref_vocab` object to be of type `LowerCaseSwahiliRV`, instead got '%s'" % (type(ref_vocab).__name__)
        
        self._rv = ref_vocab
        self.unk_char = unk_char_token
        self.oos_token = OOS_TOKEN
        self.mask_token = MASK_TOKEN
        self.pref_token = PREF_TOKEN
        self.tk_trsf = token_transformer
        
        if self._rv is not None:
            self.chars = self._rv.base_characters
            self.nums = self._rv.base_numbers
            self.nl_chars = self._rv.base_word_non_letter_chars # These are for the characters which arent letters 
            self.puncts = self._rv.base_punctuations
            
        self._char2ix = {}
        self._ix2char = None # this wil be generated only when needed
        
        self._initialize()
        
    def transform(self, string: str):
        return self.tk_trsf(string)
    
    @property
    def special_chars(self):
        return [self.unk_char, self.oos_token, self.mask_token, self.pref_token]
        
    def _initialize(self):
        """Build the required things from the get-go"""
        
        # adding spect
        characters_iters_to_build = (self.special_chars, self.chars, self.nl_chars, self.nums, self.puncts)
        
        # build the character to number dict
        for ch in chain.from_iterable(characters_iters_to_build):
            if ch not in self._char2ix:
                self._char2ix[ch] = len(self._char2ix)
                
        # TODO:
        #  think of adding a component that 
        #  represents an unknown character
        
    @property
    def char2ix(self):
        return self._char2ix
    
    @property
    def ix2char(self):
        if self._ix2char is None:
            # built the ix to char map
            self._ix2char = { ix: ch for ch, ix in self.char2ix.items() }
            
        return self._ix2char
    
    def get_index(self, char: str, strict: bool=False):
        try:
            return self.char2ix[char]
        except KeyError:
            if strict:
                raise KeyError("The character '%s' is not present in the alphabet list" % (char))
            else:
                return self.get_index(self.unk_char)
            
    def mask_characters(self, text: str, subtext_to_mask: str, coded=True):
        """Convert the text"""
    
        if len(text) == 0:
            if not coded:
                return text
            else:
                return self.encode(text)
        
        inx = text.find(subtext_to_mask)
        
        if inx != -1:
            einx = inx + len(subtext_to_mask)
            left, _, right = text[:inx], text[inx: einx], text[einx:]
            
            if not coded: 
                # actual masking
                masked = self.mask_token * len(subtext_to_mask)
            else:
                masked = [self.get_index(self.mask_token)] * len(subtext_to_mask)
                
            
            return self.mask_characters(left, subtext_to_mask, coded=coded) + masked + self.mask_characters(right, subtext_to_mask, coded=coded)
            
        
        if not coded:
            return text
        else:
            return self.encode(text)
    
    def get_char(self, ix: int, strict: bool=False):
        try:
            return self.ix2char[ix]
        except KeyError:
            if strict:
                raise KeyError("The index '%s' is not present in the indices list" % (ix))
            else:
                return self.oos_token
    
    def encode(self, string: str, strict=False):
        string = self.transform(string)
        # check if there are any special characters
        return [self.get_index(i, strict=strict) for i in string]
    
    def decode(self, int_seq: List[int], strict=False):
        return "".join([self.get_char(i) for i in int_seq])
    
    def __getitem__(self, char: str):
        return self.get_index(char)