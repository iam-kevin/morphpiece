UNK_TOKEN, REGEX_UNK_TOKEN = '[UNK]', r'\[UNK\]'
UNK_CHAR, REGEX_UNK_CHAR = '[UNKC]', r'\[UNKC\]'
NUM_TOKEN, REGEX_NUM_TOKEN = '[NUM]', r'\[NUM\]'

EOS_TOKEN, REGEX_EOS_TOKEN = '[EOS]', r'\[EOS\]'
BOS_TOKEN, REGEX_BOS_TOKEN = '[BOS]', r'\[BOS\]'
PAD_TOKEN, REGEX_PAD_TOKEN = '[PAD]', r'\[PAD\]'


class _ReferenceVocabulary(object):
    pass


class LowercaseSwahiliReferenceVocabulary(_ReferenceVocabulary):
    """Banks the parts of words that make up swahili words
    This is to include the characters that make us valid swahili words
    Things to note:
    Normal Swahili words (includes):
    letters: a-z
    other characters: dash(-), apostrophe(')
    Other words (acceptable in swahili):
    letters:
    """

    # this number regex applies for currency, decimals, and normal numbers
    # modified version from:
    #   https://stackoverflow.com/questions/5917082/regular-expression-to-match-numbers-with-or-without-commas-and-decimals-in-text
    regex_for_numbers = r'(?<!\S)(?=.)(0|([1-9](\d*|\d{0,2}(,\d{3})*)))?(\.\d*[1-9])?(?!\S)|(\d+)'

    # list of allowed characters
    base_characters = 'abcdefghijklmnoprstuvwyz'
    base_numbers = '0123456789'
    base_word_non_letter_chars = '\'-'
    base_punctuations = '.,?!()%&/:[]'  # important to include '[' and ']'

    special_tokens = [UNK_TOKEN, EOS_TOKEN, BOS_TOKEN, NUM_TOKEN, UNK_CHAR]
    regex_special_tokens = [REGEX_UNK_TOKEN, REGEX_EOS_TOKEN, REGEX_BOS_TOKEN, REGEX_NUM_TOKEN, REGEX_UNK_CHAR]

    def _backslash_punctuations(self):
        return [f'\\{c}' for c in self.base_punctuations]

    @property
    def regex_non_word(self):
        return "".join(self._backslash_punctuations())

    @property
    def regex_word(self):
        return r'{}{}{}'.format(self.base_characters, self.base_numbers, self.base_word_non_letter_chars)

    @property
    def inverse_regex_non_word(self):
        non_word = self._backslash_punctuations() + [r'\s+']
        return r'((?![{}{}]+)\W)'.format(self.base_word_non_letter_chars, "".join(non_word))

    @property
    def inverse_regex_word(self):
        return r'((?![{}]+)[\w{}]+)'.format(self.regex_word, self.base_word_non_letter_chars)

    def get_all_characters(self):
        return self.special_tokens + \
               list(self.base_characters) + \
               list(self.base_word_non_letter_chars) + \
               list(self.base_numbers) + \
               list(self.base_punctuations)