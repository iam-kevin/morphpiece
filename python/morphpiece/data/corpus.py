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

class TokenTransformer:
    def transform(self, token: str) -> Any:
        raise NotImplementedError()
    
    def __call__(self, *args, **kwargs):
        return self.transform(*args, **kwargs)
    
class BasicTokenTransformer(TokenTransformer):
    def __init__(self, lower=True):
        self._islower = lower
            
    def transform(self, token: str):
        if self._islower:
            return token.lower()
        
        return token    

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
