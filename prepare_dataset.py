#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : Oct-06-22 18:12
# @Author  : Kan HUANG (kan.huang@connect.ust.hk)


import os
import json
from pathlib import Path
from utils.bio_decode import bio_decode, load_bio_sentences, save_bio_dataset


def load_bio_dataset(bio_dir, dataset_name="MSRA dataset"):

    # Check that the dataset exist, two balnk lines at the end of the file
    path_train_val = os.path.join(bio_dir, "train.tsv")
    path_test = os.path.join(bio_dir, "test.tsv")
    msg = '{} or {} file not found. Make sure you have downloaded the right dataset'.format(
        path_train_val, path_test)
    assert os.path.isfile(path_train_val) and os.path.isfile(path_test), msg

    # Load the dataset into memory
    print(f'Loading {dataset_name} into memory...')
    dataset_train_val = load_bio_sentences(path_train_val)
    dataset_test = load_bio_sentences(path_test)
    print('- done.')

    dataset_splits = {"train": dataset_train_val, "test": dataset_test}

    return dataset_splits


def to_doccano_format():

    msra_bio_dir = Path("data/MSRA/BIO")
    dataset_splits = load_bio_dataset(msra_bio_dir)

    msra_doccano_dir = Path("data/MSRA/doccano")
    global_item_id = 0
    global_entity_id = 0
    for phase in ["train", "test"]:
        new_filepath = os.path.join(msra_doccano_dir, f"{phase}.jsonl")
        print(f'Writing dataset_split {phase} into file: {new_filepath}...')
        with open(new_filepath, 'w', encoding="utf-8") as output_file:
            for (words, tags) in dataset_splits[phase]:
                global_item_id += 1
                item_id = global_item_id
                # text = ' '.join(words)
                text = ''.join(words)
                entities = bio_decode(words, tags)
                new_entities = []
                for entity in entities:
                    global_entity_id += 1
                    new_entities.append({
                        "entity": entity.entity,
                        "id": global_entity_id,
                        "label": entity.tag,
                        "start_offset": entity.begin - 1,
                        "end_offset": entity.end - 1
                    })

                item = {"id": item_id, "text": text,
                        "entities": new_entities, "relations": []}
                output_file.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    to_doccano_format()


if __name__ == "__main__":
    main()
