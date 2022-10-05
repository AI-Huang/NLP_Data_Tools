#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-05-22 16:53
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

# File: bio_decode.py
# Port from: https://github.com/lemonhu/NER-BERT-pytorch/blob/master/build_msra_dataset_tags.py
# Refer to: https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/utils/bmes_decode.py

import os
from typing import Tuple, List


class Entity(object):
    def __init__(self, entity, tag, begin, end):
        self.entity = entity
        self.tag = tag
        self.begin = begin
        self.end = end

    def to_tuple(self):
        return tuple([self.entity, self.begin, self.end])

    def __str__(self):
        return str({key: value for key, value in self.__dict__.items()})

    def __repr__(self):
        return str({key: value for key, value in self.__dict__.items()})


def load_bio_dataset(bio_filepath) -> List[Tuple[str, str]]:
    """Load BIO dataset into memory from a text file.
    Inputs:
        bio_filepath: a data file with lines:
            当	O
            希	O
            望	O
            工	O
            程	O
            ...
    Returns:
        dataset: List[Tuple[str, str]], a BIO dataset, where each tuple consists of a complete sentence delimilated by spaces, and corresponding BIO tags, for example: 
        (
            ['当', '希', '望', '工', '程', '救', '助', '的', '百', '万', '儿', '童', '成', '长', '起', '来', '，', '科', '教', '兴', '国', '蔚', '然', '成', '风', '时', '，', '今', '天', '有', '收', '藏', '价', '值', '的', '书', '你', '没', '买', '，', '明', '日', '就', '叫', '你', '悔', '不', '当', '初', '！'],
            ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
        ).
    """
    dataset = []
    with open(bio_filepath) as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for line in f:
            if line != '\n':
                line = line.strip('\n')
                word, tag = line.split('\t')
                try:
                    # In case that word == '0' and tag == '\t'
                    if len(word) > 0 and len(tag) > 0:
                        word, tag = str(word), str(tag)
                        words.append(word)
                        tags.append(tag)
                except Exception as e:
                    print('An exception was raised, skipping a word: {}'.format(e))
            else:
                if len(words) > 0:
                    assert len(words) == len(tags)
                    dataset.append((words, tags))
                    words, tags = [], []

    return dataset


def save_bio_dataset(dataset, save_dir):
    """Write sentences.txt and tags.txt files in save_dir from BIO dataset.
    Args:
        dataset: List[Tuple[str, str]], a BIO dataset, e.g., 
            ([(["a", "cat"], ["O", "O"]), ...]).
        save_dir: (string)
    """
    # Create directory if it doesn't exist
    print('Saving in {}...'.format(save_dir))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Export the dataset
    with open(os.path.join(save_dir, 'sentences.txt'), 'w') as file_sentences, \
            open(os.path.join(save_dir, 'tags.txt'), 'w') as file_tags:
        for words, tags in dataset:
            file_sentences.write('{}\n'.format(' '.join(words)))
            file_tags.write('{}\n'.format(' '.join(tags)))
    print('- done.')


def bio_decode(words, tags):
    """decode BIO inputs to tags
    Inputs:
        char_label_list: list of tuple (word, bmes-tag)
    Returns:
        tags
    Examples:
        >>> x = [("Hi", "O"), ("Beijing", "B-LOC")]
        >>> bio_decode(x)
        [{'entity': 'Beijing', 'tag': 'LOC', 'begin': 1, 'end': 2}]
    """
    length = len(words)
    assert length >= 2
    assert len(words) == len(tags)

    idx = 1
    entities = []

    # State machine transition cases:
    # O -> O, do nothing; O -> I, impossible,
    # O -> B, B -> B, I -> B, start labeling entity,
    # B -> I, I -> I, keep labeling,
    # B -> O, I -> O, stop labeling, reset entity.

    entity, start_position, end_position = "", None, None
    previous_word, previous_tag = words[idx], tags[idx]
    previous_label = previous_tag[0]
    while idx < length:
        current_word, current_tag = words[idx], tags[idx]
        current_label = current_tag[0]

        # Check if to stop previous labeling
        if current_label == "O" or current_label == "B":
            if entity != "":
                entities.append(
                    Entity(entity, previous_tag[2:], start_position, end_position))
                entity, start_position, end_position = "", None, None

        if current_label == "B":
            start_position, end_position = idx, idx + 1
            entity += current_word
        elif current_label == "I":
            # Uncomment to throw entities labeled starting with I-XXX
            # if start_position is None:
            #     idx += 1
            #     continue
            # In case that labels start with I-XXX
            start_position = idx if start_position is None else start_position
            end_position = end_position+1 if end_position is not None else idx+1
            entity += current_word
        elif current_label == "O":
            pass

        previous_word, previous_tag = words[idx], tags[idx]
        previous_label = previous_tag[0]
        idx += 1

    return entities


def main():
    msra_bio_dir = "data/msra_bio"
    msra_mrc_dir = "data/msra_mrc"
    tag2query_file = "ner2mrc/queries/zh_msra.json"
    os.makedirs(msra_mrc_dir, exist_ok=True)

    # Check that the dataset exist, two balnk lines at the end of the file
    path_train_val = os.path.join(msra_bio_dir, "train.tsv")
    path_test = os.path.join(msra_bio_dir, "test.tsv")
    msg = '{} or {} file not found. Make sure you have downloaded the right dataset'.format(
        path_train_val, path_test)
    assert os.path.isfile(path_train_val) and os.path.isfile(path_test), msg

    # Load the dataset into memory
    print('Loading MSRA dataset into memory...')
    dataset_train_val = load_bio_dataset(path_train_val)
    dataset_test = load_bio_dataset(path_test)
    print('- done.')

    # # Make a list that decides the order in which we go over the data
    # order = list(range(len(dataset_train_val)))
    # # random.seed(2019)
    # # random.shuffle(order)

    # # Split the dataset into train, val(split with shuffle) and test
    # train_dataset = [dataset_train_val[idx]
    #                  for idx in order[:42000]]  # 42000 for train
    # val_dataset = [dataset_train_val[idx]
    #                for idx in order[42000:]]  # 3000 for val
    # test_dataset = dataset_test  # 3442 for test

    # save_bio_dataset(train_dataset, 'data/msra/train')
    # save_bio_dataset(val_dataset, 'data/msra/val')
    # save_bio_dataset(test_dataset, 'data/msra/test')

    save_bio_dataset(dataset_train_val, 'data/msra/train_val')
    save_bio_dataset(dataset_test, 'data/msra/test')


if __name__ == '__main__':
    main()
