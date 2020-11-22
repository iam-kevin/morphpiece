# MorphPiece

[![CircleCI](https://circleci.com/gh/iam-kevin/morphpiece/tree/main.svg?style=svg&circle-token=5c1789af0f217b956fcadec6bb2787bd7135035e)](https://app.circleci.com/pipelines/github/iam-kevin/morphpiece/6/workflows/a7bffeba-b31d-4e72-80c6-4118211c97d1)
[![GitHub tag (latest SemVer pre-release)](https://img.shields.io/github/v/tag/iam-kevin/morphpiece??style=flat&logo=appveyor&include_prereleases&sort=semver)](https://github.com/iam-kevin/morphpiece/tree/main)

MorphPiece is _an attempt to build_ a semi-supervised / unsupervised learning text tokenizer and detokenizer that takes a more linguistic approach of composition. The learning technique is attributed from the paper on [Linguistica](http://people.cs.uchicago.edu/~jagoldsm/linguistica-site/), as well as using partial computation of cost matrix inspired by the [Levenstien distance](https://en.wikipedia.org/wiki/Levenshtein_distance) algorithm

The name is creatively inspired by WordPiece and SentencePiece, since it tries to achieve the same goal, but only through learning the morpheme-like subword units of the different words.

<!-- **This is a product used by the Inspired Ideas Reseach Team** -->

## Overview

### What is MorphPiece?

MorphPiece _would be_ a re-implementation of sub-word units that is based of collecting morpheme-like subword units using a cost matrix technique. The end-goal for building such a tokenizer, like the SentencePiece and WordPiece, is that, it would break words into its sub-word parts, but unlike those 2 implementation, it would determine based on the concept of _morphemes_ and giving precedence to breaking words from **agglunative languages** like Swahili *(What the model is built around)*

### Features

#### Works with any number of texts

#### Configurations for composition learning is pre-determined

#### Able to set existing morphemes

To provide the learning with prior knowledge, we are able to set the list of morphemes that we expect the tokenizer to learn from. 

Looking at the version below

**Without the morpheme list**
```python
from morphpiece.trainer import MorphPieceTrainer
from morphpiece.processor import AlphabetProcessor
from morphpiece.data import WordList
from morphpiece.core import MorphPiece

mpt = MorphPieceTrainer(alphabet_processor=AlphabetProcessor('swahili'))
mp: MorphPiece = mpt.train(
    WordList.from_corpus('swahili-corpus.txt')
)
print(mp)
>>> MorphPiece(count=3423, morphs=['anz', 'end', ...])
mp.tokenize('anapenda', coded=False)
>>> ['_a', 'na', 'p', 'end', 'a']
```

**With the morpheme list**
```python
...

mpt = MorphPieceTrainer(alphabet_processor=AlphabetProcessor('swahili'))
mp: MorphPiece = mpt.train(
    WordList.from_corpus('swahili-corpus.txt'), 
    morphemes=['end', 'pend'] # Including the wordlist indicator
)
print(mp)
>>> MorphPiece(count=3423, morphs=['anz', 'end', 'pend', ...])
mp.tokenize('anapenda', coded=False)
>>> ['_a', 'na', 'pend', 'a']

```

## Installation

Installing morphpiece using a `pip` command

```bash
pip install morphpiece
```

## Usage

### Obtain from Training

```python
from morphpiece.trainer import MorphPieceTrainer
from morphpiece.processor import AlphabetProcessor

mpt = MorphPieceTrainer(alphabet_processor=AlphabetProcessor('swahili'))
mp = mpt.train(corpus_path='sw-corpus.txt', ...)
```

### Obtain from pre-trained morphpiece
<!-- TODO: revise this pre-trained morphpiece -->

```python
from morphpiece.core import MorphPiece
mp = MorphPiece.load('mp-sw-model.mdl')
```

