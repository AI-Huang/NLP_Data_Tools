#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-05-22 16:53
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)

# File: bio_decode.py
# Port from: https://github.com/lemonhu/NER-BERT-pytorch/blob/master/build_msra_dataset_tags.py
# Refer to: https://github.com/ShannonAI/mrc-for-flat-nested-ner/blob/master/utils/bmes_decode.py

import os
import json
from pathlib import Path
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


def load_bio_sentences(bio_filepath) -> List[Tuple[str, str]]:
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
    with open(bio_filepath, encoding="utf-8") as f:
        words, tags = [], []
        # Each line of the file corresponds to one word and tag
        for i, line in enumerate(f):
            if line != '\n':
                line = line.strip('\n')
                word, tag = line.split('\t')
                try:
                    if len(word) > 0 and len(tag) > 0:
                        pass
                    # In case that word == '0' and tag == ''
                    else:
                        # print(i, line)
                        # Correct the tag
                        tag = 'O'
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
    print(f"Saving in \"{save_dir}\"...")
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

    entities = []

    # State machine transition cases:
    # O -> O, do nothing; O -> I, impossible,
    # O -> B, B -> B, I -> B, start labeling entity,
    # B -> I, I -> I, keep labeling,
    # B -> O, I -> O, stop labeling, reset entity.

    entity, start_position, end_position = "", None, None
    previous_word, previous_tag = words[0], tags[0]
    previous_label = previous_tag[0]

    idx = 1
    while idx < length:
        # Process past labels
        if previous_label == "B":
            start_position, end_position = idx, idx + 1
            entity += previous_word
        elif previous_label == "I":
            # Uncomment to throw entities labeled starting with I-XXX
            # if start_position is None:
            #     idx += 1
            #     continue
            # In case that labels start with I-XXX
            start_position = idx if start_position is None else start_position
            end_position = end_position+1 if end_position is not None else idx+1
            entity += previous_word
        elif previous_label == "O":
            pass

        # Process current label
        current_word, current_tag = words[idx], tags[idx]
        current_label = current_tag[0]

        # Check whether to stop previous labeling
        if current_label == "O" or current_label == "B":
            if entity != "":
                entities.append(
                    Entity(entity, previous_tag[2:], start_position, end_position))
                entity, start_position, end_position = "", None, None

        previous_word, previous_tag = words[idx], tags[idx]
        previous_label = previous_tag[0]
        idx += 1

    return entities


def get_sent_num_tags(words, tags, tag_type):
    """
    Inputs:
        words:
        tags:
        tag_type: e.g., "LOC".
    """
    num_tags_gt = 0
    for tag in tags:
        if tag.startswith('B'):
            if tag[2:] == tag_type:
                num_tags_gt += 1

    num_tags = 0
    entities = bio_decode(words, tags)
    for entity in entities:
        if entity.tag == tag_type:
            num_tags += 1

    return num_tags_gt, num_tags


def bio_decode_test(dataset_splits, tag_nums):
    """
    Inputs:
        dataset_splits: 
        tag_nums: dictionary, e.g., for MSRA dataset,

        tag_nums = {
            "train": {"LOC": 36860, "ORG": 20584, "PER": 17615},
            "test": {"LOC": 2886, "ORG": 1331, "PER": 1973}
        }
    """

    print('Start testing BIO decoding...')

    for phase in ["train", "test"]:
        print(f"Testing phase: {phase}.")
        for test_tag_type in tag_nums[phase].keys():
            all_num_tags_gt = 0
            for (words, tags) in dataset_splits[phase]:
                num_tags_gt, num_tags = get_sent_num_tags(
                    words, tags, tag_type=test_tag_type)
                # Validate each sentence's number of tags
                valid = True if num_tags == num_tags_gt else False
                if not valid:
                    raise ValueError("num_tags and num_tags_gt are NOT equal.")
                all_num_tags_gt += num_tags_gt

            # Validate the number of `test_tag_type` among all the sentences
            if all_num_tags_gt != tag_nums[phase][test_tag_type]:
                raise ValueError(
                    f"all_num_tags_gt ({all_num_tags_gt}) NOT correct, true number: {tag_nums[phase][test_tag_type]}.")

    print('Test done.')


def main():
    dataset_name = "MSRA"  # "MSRA", "People's Daily"
    bio_dir = Path(f"data/{dataset_name}/BIO")
    dataset_files = json.load(
        open(os.path.join(bio_dir, "dataset_files.json")))

    phases = dataset_files.keys()  # "train", "test"
    # Check that the dataset exist, two blank lines at the end of the file
    for phase in phases:
        dataset_files[phase] = os.path.join(bio_dir, dataset_files[phase])
        msg = f"{dataset_files[phase]} file not found. Make sure you have downloaded the right dataset"
        assert os.path.isfile(dataset_files[phase]), msg

    # Load the dataset into memory
    print(f'Loading {dataset_name} dataset into memory...')
    dataset_splits = {}
    for phase in phases:
        dataset_splits[phase] = load_bio_sentences(dataset_files[phase])
    print('- done.')

    tag_nums = json.load(
        open(os.path.join(bio_dir, "tag_nums.json")))
    bio_decode_test(dataset_splits, tag_nums)

    # # Make a list that decides the order in which we go over the data
    # order = list(range(len(dataset_train_val)))
    # random.seed(2019)
    # random.shuffle(order)

    # # Split the dataset into train, val(split with shuffle) and test
    # train_dataset = [dataset_train_val[idx]
    #                  for idx in order[:42000]]  # 42000 for train
    # val_dataset = [dataset_train_val[idx]
    #                for idx in order[42000:]]  # 3000 for val
    # test_dataset = dataset_test  # 3442 for test

    # save_bio_dataset(train_dataset, 'data/msra/train')
    # save_bio_dataset(val_dataset, 'data/msra/val')
    # save_bio_dataset(test_dataset, 'data/msra/test')

    sent_dir = Path(f"data/{dataset_name}/sentences")
    for phase in phases:
        save_bio_dataset(dataset_splits[phase], os.path.join(sent_dir, phase))


if __name__ == '__main__':
    main()
