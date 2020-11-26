import os
import re

from pathlib import Path

import torch
from torch.utils.data import Dataset, ConcatDataset

from typing import Tuple, List, Iterable, Any, Union

from ..tokenizer import MutableTokenizer, Tokenizer
from .corpus import DataTextTransformer, WordList, LazyDataTextReader
from ..utils.data import generate_context_samples

from itertools import chain


class MorphDataset(Dataset):
    """Morphology Dataset"""
    def __init__(self, data: Iterable[Tuple[str, List[str]]], tokenizer: Tokenizer):
        self.ctx_data = data
        self.tkr = tokenizer
    
    def __getitem__(self, ix: int):
        inp, ctxs = self.ctx_data[ix]
        
        inp_ix = torch.tensor(self.tkr.encode(inp))
        ctxs_ix = list(map(lambda x: torch.tensor(self.tkr.encode(x)), ctxs))

        return inp_ix, ctxs_ix
    
    def __len__(self):
        return len(self.ctx_data)


class DataBuilder:
    def __init__(self,
                 sentence_transformer: DataTextTransformer,
                 window_size: int = 5, 
                 neg_samples = 10):
        self.ws = window_size
        self.ns = neg_samples
        self.sentence_tokenize = sentence_transformer
    
    def build_from_sentences(self, sentences: List[str], *args, **kwargs):
        pass
    
    def build_from_sentence(self, sentence: str, *args, **kwargs):
        tk = self.sentence_tokenize(sentence)
        return self.build_from_tokens(tk, *args, **kwargs)
    
    def build_from_tokens(self, token_seq: List[Any], *args, **kwargs):
        return list(generate_context_samples(token_seq, self.ws, *args, **kwargs))
    
    def build_from_file(self, file_path: Union[str, os.PathLike], *args, **kwargs):
        """Working with Path"""
        file_path = Path(file_path)
        
        assert file_path.exists(), "No such file '%s'" % (str(file_path))
        return chain.from_iterable(self._generate_contexted_words_from_file(file_path, *args, **kwargs))
    
    def _generate_contexted_words_from_file(self, file_path, *args, **kwargs):
        # read the corpus data
        reader = LazyDataTextReader(transformer=self.sentence_tokenize)
        for token_seq in (reader.read(file_path)):
            yield self.build_from_tokens(token_seq, *args, **kwargs)
            
            # TODO: if you plan on removing this, simply change the 
            #  then change the `chain.from_iterable` to something else
            # yield [SEP_TOKEN]

    def build_dataset(self, file_path: str, tokenizer: MutableTokenizer, *args, **kwargs):
        ctx_data = [MorphDataset(c, tokenizer=tokenizer) \
                        for c in self._generate_contexted_words_from_file(file_path, *args, **kwargs)]

        return ConcatDataset(ctx_data)