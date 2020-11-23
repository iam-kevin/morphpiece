from unicodedata import normalize as str_normalize
from .data import TokenTransformer, WordList
from .processor import AlphabetProcessor

def unicode_normalize(text: str, technique: str = 'NFKC'):
    return str_normalize(technique, text)

class MorphPiece:
    def __init__(self, 
                 start_morphs: WordList, 
                 inner_morphs: WordList, 
                 alphabet_processor: AlphabetProcessor,
                 tokenTransformer: TokenTransformer = None):
        
        self._imorphs = start_morphs
        self._morphs = inner_morphs
        
        self.alpp = alphabet_processor
        self._tktrsf = tokenTransformer
        
    def _clean_transform(self, text: str):
        if self._tktrsf is not None:
            return self._tktrsf(text)
        
        return text
    
    def tokenize_word(self, word: str):
        """When encoding, add the index for _morphs, by len(_imorphs)"""
        word = self._clean_transform(word)
        
        # morpheme pieces
        pieces = []
        
        for pmorp in self._imorphs.get_longest_first_list(True):
            if word.find(pmorp) == 0:
                pieces.append(self.alpp.pref_token + pmorp)
                
                # update the word
                word = word[len(pmorp):]
                break
                
        # when there are no pieces broken from the words
        if len(pieces) == 0:
            return [self.alpp.pref_token + word]
                
        return pieces + self._break_base_word(word)
                
        
    def _break_base_word(self, word: str):
        for morp in self._morphs.get_longest_first_list():
            if word.find(morp) == 0:
                return [morp] + self._break_base_word(word[len(morp):])
            
        return [word]
