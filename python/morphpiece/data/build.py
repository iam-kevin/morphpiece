import os
import re

from pathlib import Path

import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

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
        key, out = self.ctx_data[ix]
        context = list(chain.from_iterable(self.tkr.encode(w) for w in out['context']))
        neg_samples = list(chain.from_iterable(self.tkr.encode(w) for w in out['neg_samples']))
        
        context = list(set(context))
        neg_samples = list(set(neg_samples))
        
        return torch.tensor(self.tkr.encode(key)), torch.tensor(context), torch.tensor(neg_samples)
#         inp, ctxs = self.ctx_data[ix]
        
#         inp_ix = torch.tensor(self.tkr.encode(inp))
#         ctxs_ix = list(map(lambda x: torch.tensor(self.tkr.encode(x)), ctxs))

#         return inp_ix, ctxs_ix

    def __len__(self):
        return len(self.ctx_data)

class DataBuilder:
    def __init__(self, sentence_tranformer: DataTextTransformer, tokenizer: MutableTokenizer, **options):
        self.gener_options = options
        self.sentence_tokenize = sentence_tranformer
        self.tokenizer = tokenizer
        
    def build_from_sentences(self, sentences: List[str], *args, **kwargs):
        pass
    
    def build_from_sentence(self, sentence: str, *args, **kwargs):
        tk = self.sentence_tokenize(sentence)
        return self.build_from_tokens(tk, *args, **kwargs)
    
    def build_from_tokens(self, token_seq: List[Any], *args, **kwargs):
        return list(generate_context_samples(token_seq, *args, **self.gener_options, **kwargs))
    
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

    def build_dataset(self, file_path: str, *args, **kwargs):
        ctx_data = [MorphDataset(c, tokenizer=self.tokenizer) \
                        for c in self._generate_contexted_words_from_file(file_path, *args, **kwargs)]

        return ConcatDataset(ctx_data)
    
    def _collate_fn(self, out):
        # transpose
        out = list(zip(*out))
        
        # pad and grow
        return [pad_sequence(o, padding_value=self.tokenizer['pad']).T for o in out]
    
    def build_dataloader(self, file_path: str, **dataloader_opts):
        dataset = self.build_dataset(file_path)

        # convert to immutable to build 'pad' token
        self.tokenizer = self.tokenizer.immutable()

        return DataLoader(dataset, collate_fn=self._collate_fn, **dataloader_opts)
