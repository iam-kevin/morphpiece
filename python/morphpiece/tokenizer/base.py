PAD_TOKEN = '[pad]'

# Unknown token
UNK_TOKEN = '[unk]'
B_UNK_TOKEN = '[b-unk]'
E_UNK_TOKEN = '[e-unk]'
FL_UNK_TOKEN = '[fl-unk]'

START_PIECE_INDICATOR = '<'
END_PIECE_INDICATOR = '>'

class Tokenizer:
    def tokenize(self, *args, **kwargs):
        raise NotImplementedError()
    
    def encode(self, *args, **kwargs):
        raise NotImplementedError()
        
    def decode(self, *args, **kwargs):
        raise NotImplementedError()


class MutableTokenizer(Tokenizer):
    def immutable(self, *args, **kwargs):
        raise NotImplementedError()


class ImmutableTokenizer(Tokenizer):
    pass
