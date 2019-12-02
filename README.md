# Metropolis-Hastings Attacker

This is the implementation of Metropolis-Hastings Attacker. Please refer to the paper "Generating Fluent Adversarial Examples for Natural Languages" (ACL2019).

## How2Use

Step 1. Please tokenize the IMDB raw data, and build the vocabulary list by frequency order. Save the processed data as the `dataset.SeqClassificationDataset` function requires.

Step 2. Train and retrieve the classifier, which is in the `classifier` package, and the language model, which is in the `lm` package.

Step 3. Generate a subset where all examples are correctly classified by the retrieved classifier.

Step 4. Run white-box (`attack_wb`) or black-box (`attaCK_bb`) MHA on the subset.

If you want to use other datasets, or your own victim classifiers or language models, please skip Step 1 and 2.

## Cite

```
@inproceedings{generating2019zhang,
  author    = {Huangzhao Zhang and
               Hao Zhou and
               Ning Miao and
               Lei Li},
  title     = {Generating Fluent Adversarial Examples for Natural Languages},
  booktitle = {Proceedings of the 57th Conference of the Association for Computational
               Linguistics, {ACL} 2019, Florence, Italy, July 28- August 2, 2019,
               Volume 1: Long Papers},
  pages     = {5564--5569},
  year      = {2019}
}
```
