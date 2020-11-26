from .reference_vocabulary import LowercaseSwahiliReferenceVocabulary, _ReferenceVocabulary

# adding Text Transformers
from .corpus import (DataTextTransformer, 
                        TokenTransformer, 
                        BasicTokenTransformer, 
                        SimpleTextTransformer)

# adding readers
from .corpus import LazyDataTextReader

# adding Dictionary-based components
from .corpus import WordList
