from .base import ImmutableTokenizer
from .base import (PAD_TOKEN, UNK_TOKEN, 
                    B_UNK_TOKEN, E_UNK_TOKEN, 
                    FL_UNK_TOKEN,
                    START_PIECE_INDICATOR, END_PIECE_INDICATOR)

from ..data.corpus import WordList

from typing import List

import logging
logger = logging.getLogger(__name__)

class SyllabicTokenizer(ImmutableTokenizer):
    def __init__(self, 
                 ls_token: List[str],
                 pad_token=PAD_TOKEN, 
                 unk_token=UNK_TOKEN, 
                 b_unk_token=B_UNK_TOKEN, 
                 e_unk_token=E_UNK_TOKEN, 
                 fl_unk_token=FL_UNK_TOKEN):
        
        # special tokens
        self._tk = {
            'pad': pad_token,
            'unk': unk_token,
            'b_unk': b_unk_token,
            'e_unk': e_unk_token,
            'fl_unk': fl_unk_token
        }
        # flagged word list
        self._fwl = WordList([list(self._tk.values()), ls_token])
        self.cached_subword_list = self._fwl.get_alphabetized_list(asc=True, widest_first=True)
        
    @property
    def special_token(self):
        return self._tk
    
    def __getitem__(self, val: str):
        """Get index from val"""
        if val in self._tk:
            val = self._tk[val]
        
        return self._fwl.get_index(val)
    
    def __len__(self):
        return len(self._fwl)
    
    def has(self, token: str):
        return token in self._fwl.token2ix
    
    def _encode(self, transformed_word: str):
        # rename variable
        word = transformed_word
        en_seq = []
        
        for syl in [i for i in self.cached_subword_list if len(word) >= len(i)]:
            if word.find(syl) == 0:
                # add syl/ to list
                en_seq.append(syl)
                
                word = word[len(syl):]
                break
                
        if len(en_seq) == 0:
            # empty string
            if word == '': return []
            
            # move to next text break
            return [self.special_token['unk']] + self._encode(word[1:])
        else:
            # If the word is decode
            return en_seq + self._encode(word)
    
    def tokenize(self, word: str, start_flag: bool=True, end_flag: bool=True):
        word = word.strip() # TODO: change this
        seq = self._encode(word)
        
        data = []
        for i, tk in enumerate(seq):
            # first
            if start_flag:
                if i == 0:
                    if tk == self.special_token['unk']:
                        data.append(self.special_token['b_unk'])
                    else:
                        _tk = START_PIECE_INDICATOR + tk
                        if self.has(_tk):
                            data.append(_tk)
                        else:
                            data.extend([START_PIECE_INDICATOR, tk])
                    continue
            else:
                if i == 0:
                    data.append(tk)
                    continue
                
            # end
            if end_flag:
                if i == len(seq) - 1:
                    if tk == self.special_token['unk']:
                        data.append(self.special_token['e_unk'])
                    else:
                        _tk = tk + END_PIECE_INDICATOR
                        if self.has(_tk):
                            data.append(_tk)
                        else:
                            data.extend([tk, END_PIECE_INDICATOR])
                    continue
            else:
                if i == len(seq) - 1:
                    data.append(tk)
                    continue
                    
            # mid
            data.append(tk)
            
        return data

    def encode(self, word: str):
        return [self[sbw] for sbw in self.tokenize(word)]
