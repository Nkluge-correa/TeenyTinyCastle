# Stanford Encyclopedia of Philosophy Text Corpus v0.1

**Stanford Encyclopedia of Philosophy Text Corpus v0.1** can be downloaded in [this link](https://drive.google.com/uc?export=download&id=167nlUxmY0Qc9_wGMuyMqgh2pUwyJDrte).

## Overview

The Stanford Encyclopedia of Philosophy ([SEP](https://plato.stanford.edu/about.html)) is a dynamic reference work, including over 1,770 entries written by top scholars in the field of philosophy.

This dataset contains the full text of all articles contained within the SEP. All data is available as a `CSV` file and a folder of `.txt` files. The CSV files possess information related to the original page (`URL`), the subject of the page (`Category`), and the text of the page (`Text`). This dataset can be used for NLP applications like text mining, text classification, and text generation, among others.

## Dataset

This dataset contains 182531 text entries related to 1,770 different philosophical subjects.

All entries can be found in the `CSV` file (`stanford_encyclopedia_philosophy.csv`), or as separate `.txt` files in the `dataset` folder. These `txt` files are separated by category (Socrates, Plato, Aesthetic) in different folders.

## Vocabulary

In addition to the text data, we also provide an already-tokenized bag of words/vocabulary of different sizes (`5000, 10000, 15000, 20000, 25000, 200000`) together with the full tokenized vocabulary (`vocab_SEP`) in the format of a `.txt` file. These vocabularies were computed using the `TextVectorization` from Keras (Tensorflow 2.10.1).
