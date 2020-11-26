from typing import List, Union, Iterable
from pathlib import Path

from ..data.corpus import WordList, BasicTokenTransformer, TokenTransformer
from tqdm import tqdm

from .base import MutableTokenizer
from .base import (START_PIECE_INDICATOR, END_PIECE_INDICATOR)

from .immutable import SyllabicTokenizer as FrozenSyllabicTokenizer

import os
import logging

# logger
logger = logging.getLogger(__name__)


class SyllabicTokenizer(MutableTokenizer):
    """Syllabic Tokenizer which can grow"""
    def __init__(self, 
                 syllables: List[str], 
                 transformer: TokenTransformer,
                 start_flag: bool=True,
                 end_flag: bool=True):
        
        self.token_transform = transformer
        self.start_flag = start_flag
        self.end_flag = end_flag
        
        self._initialize(syllables)
        
    def _initialize(self, syllables: List[str]):
        """Initializing the rest of the components properly"""
        
        indicators = [START_PIECE_INDICATOR, END_PIECE_INDICATOR]
        self._wl = WordList([indicators], token_transformer=self.token_transform)
        self._wl.add_list(syllables)
        
        # Building the word list for flagged words
        self._flwl = WordList()
        
        self.cached_syllables_list = self._wl.get_alphabetized_list(asc=True, widest_first=True)
        
    def _encode(self, transformed_word: str):
        # rename variable
        word = transformed_word
        
        en_seq = []
        for syl in [i for i in self.cached_syllables_list if len(word) >= len(i)]:
            if word.find(syl) == 0:
                # add syl/ to list
                en_seq.append(self._wl.get_index(syl))
                
                word = word[len(syl):]
                break
                
        if len(en_seq) == 0:
            # empty string
            if word == '': return []
            
            # add to list
            nc = word[0]
            self._wl.add(nc)

            # move to next text break
            return [self._wl.get_index(nc)] + self._encode(word[1:])
        else:
            # If the word is decode
            return en_seq + self._encode(word)
        
    def encode(self, word: str):
        encs = self.tokenize(word)
        
        # add the sub word tokens
        for dd in encs:
            self._flwl.add(dd)
            
        return [self._flwl.get_index(sbw) for sbw in encs]
            
    def tokenize(self, word: str, start_flag: bool=True, end_flag: bool=True):
        word = self.token_transform(word)
        seq = self._encode(word)
        
        data = [self._wl.ix2token[i] for i in seq]
        
        if start_flag:
            data[0] = START_PIECE_INDICATOR + data[0]
            
        if end_flag:
            data[-1] = data[-1] + END_PIECE_INDICATOR
            
        return data
    
    def build_from_iter(self, iter_str: Iterable[List[str]]):
        logger.info('Building the corpus of data')

        for seq in tqdm(iter_str):
            self._build_from_list(seq)
    
    def _build_from_list(self, list_str: List[str]):
        for word in list_str:
            sub_words = self.tokenize(word, start_flag=self.start_flag, end_flag=self.end_flag)
            self._flwl.add_list(sub_words)
            
    def immutable(self, sort=True):
        # build the data
        _lls = self._wl.get_words() + self._flwl.get_words()
        
        if sort:
            _lls = sorted(sorted(_lls), key=len)
        
        return FrozenSyllabicTokenizer(_lls)
    
    @classmethod
    def from_file(cls, file_path: Union[str, os.PathLike], *args, **kwargs):
        file_path = Path(file_path)
    
        assert file_path.exists(), "The syllables file doesn't exist"
        
        syl_list = None
        with open(file_path, mode='r', encoding='utf-8') as fl:
            syl_list = [line.strip() for line in fl.readlines()]
            
        return cls(syllables=syl_list, *args, **kwargs)
