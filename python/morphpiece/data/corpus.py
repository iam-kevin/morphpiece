from typing import List, Iterable, Union, Dict, Any
import os
from itertools import chain
import re


from pathlib import Path
from typing import Union
import logging

logger = logging.getLogger('morphpiece')

"""
Text Transformer
"""
class DataTextTransformer:
    def transform(self, text: str) -> Any:
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)

class TokenTransformer(DataTextTransformer):
    def transform(self, token: str) -> Any:
        raise NotImplementedError()
    
class BasicTokenTransformer(TokenTransformer):
    def __init__(self, lower=True):
        self._islower = lower
            
    def transform(self, token: str):
        if self._islower:
            return token.lower()
        
        return token

class SimpleTextTransformer(DataTextTransformer):
    def transform(self, text: str):
        text = text.strip()
        texts = re.split(r'\s+', text)
        return texts


"""
Reader
"""
class DataTextReader(object):
    def __init__(self, transformers: List[DataTextTransformer] = None):
        self.stacked_transforms = transformers

        if not (transformers is None or len(transformers) == 0):
            logger.info('Included transformers')
            for ic, transfn in enumerate(transformers):
                assert isinstance(transfn, DataTextTransformer), \
                    "Object in the transformer must implement the {}, instead got {}".format(
                        DataTextTransformer.__name__, type(transfn).__name__)

                logger.info('[Tf .{:03d}] {}'.format(ic + 1, str(transfn)))

    def read(self, file_path: Union[str, os.PathLike]):
        logger.info('Reading the file \'{}\''.format(file_path))

        # perform checks here and there

        # TODO: think of wrapping this in a _read, when implemented by someone
        with open(file_path, 'r', encoding='utf-8') as rfl:
            return [self.transform(line) for line in rfl.readlines()]

    def transform(self, text: str) -> str:
        transformed_text = text

        if not (self.stacked_transforms is None or len(self.stacked_transforms) == 0):
            for transfn in self.stacked_transforms:
                transformed_text = transfn(transformed_text)

        return transformed_text


class LazyDataTextReader(DataTextReader):
    def __init__(self, transformer: DataTextTransformer = None):
        super().__init__([transformer] if transformer is not None else None)
        
    def read(self, file_path: Union[str, os.PathLike]):
        file_path = Path(file_path)
        
        assert file_path.exists(), "The file doesn't exist"
        logger.info('Reading the file \'%s\'' % (file_path))

        with open(file_path, mode="r", encoding="utf-8") as lfl:
            for line in lfl:
                yield self.transform(line)


"""
WordList
"""

class WordList:
    """This is a component that contains all the words that 
    need to exist for some language operation"""
    def __init__(self, doc_list: Iterable[List[str]] = None, token_transformer: TokenTransformer = None):   
        self._tk_trsf = token_transformer
        
        # for caching
        self._token2ix = {}
        self._ix2token = None
        self._ixCounter = {}
        
        if doc_list is not None:
            # building wordlist
            self._initialize(doc_list)
        
    @property
    def token2ix(self):
        return self._token2ix
    
    @property
    def ix2token(self):
        if self._ix2token is None or (len(self._ix2token) != len(self._token2ix)):
            self._ix2token = { ix: tk for tk, ix in self.token2ix.items() }
            
        return self._ix2token
    
    def _token_transform(self, token: str):
        if self._tk_trsf is not None:
            return self._tk_trsf(token)
        
        return token
    
    def get_index(self, token: str):
        token = self._token_transform(token)
        
        if token in self._token2ix:
            return self._token2ix[token]
        
        raise KeyError("The token '%s' doesn't exist in the word list")
        
    def get_word(self, ix: int):
        return self.ix2token[ix]
        
    def add(self, token: str, skip_count: bool = False):
        if len(token) == 0:
            return
        
        # add word if not there
        if token not in self.token2ix:
            self._token2ix[token] = len(self.token2ix) 
            
        # add the word count
        ix = self.get_index(token)
        
        if not skip_count:
            if ix not in self._ixCounter:
                self._ixCounter[ix] = 0

            self._ixCounter[ix] += 1

    def add_list(self, words: List[str], **kwargs):
        for word in words:
            self.add(word, **kwargs)
            
    
    def get_words(self):
        return list(self.token2ix.keys())
            
    def get_alphabetized_list(self, asc=True, widest_first: Union[None, bool] = None):
        """Get the list of words sorted in alphabetical order"""
        _wl = sorted(self.get_words())
        
        if not asc:
            _wl = _wl[::-1]
        
        if widest_first is not None:
            _wl = sorted(_wl, key=len)
            
            if widest_first:
                _wl = _wl[::-1]                
        
        return _wl
    
    def get_longest_first_list(self, reverse=False):
        """Sort the list by geting the longest text first"""
        sls = sorted(self.get_words(), key=len)
        
        if not reverse:
            return sls[::-1]

        return sls
    
    @property
    def token_count_by_id(self):
        return self._ixCounter
    
    @property
    def token_count(self):
        return { self.get_word(ix): ct for ix, ct in self._ixCounter.items() }
    
    def __len__(self):
        return len(self.token2ix)    
    
    def __iter__(self):
        for word in self.token2ix:
            yield word
            
    def filter(self, key):
        """This is used to filter the contents in the Wordlist object
        
        TODO(@iam-kevin): Add some filtration logic here such that 
        the object returned works for the algorithm"""
        pass
            
    def _initialize(self, doc_list: Iterable[List[str]]):
        # TODO: you might want to enforce rule to make sure that the data is 
        #   Iterable[List[str]] -> like [['yeye', 'anacheza']]
        #  as opposed to 
        #   Iterable[str] -> like ['yeye', 'anacheza']
        
        for token in chain.from_iterable(doc_list):
            token = self._token_transform(token)
            
            # Add the token to dict, if doesn't exist
            # ---------------------------
            self.add(token)
        
    @classmethod
    def from_corpus(cls, file_path: Union[str, os.PathLike], transformer: DataTextTransformer = None, **kwargs):
        reader = LazyDataTextReader(transformer)
        return cls(list(reader.read(file_path)), **kwargs)
    
    @classmethod
    def from_wordlist_file(cls, wordlist_file: Union[str, os.PathLike], **kwargs):
        # ... using the same logic here
        pass
    
    @classmethod
    def from_wordmap(cls, wordmap: Dict[str, int], **kwargs):
        # ... using the same logic here
        pass
    
    @classmethod
    def from_list(cls, ls: List[str], *args, **kwargs):
        return cls(doc_list=[ls], *args, **kwargs)
       
