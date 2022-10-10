## People's Daily(人民日报) dataset

### Data Sources

Download **original** MSRA data from: https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/People's%20Daily, then:

```bash
mkdir -p "./data/People's Daily/BIO"
mv example.train "./data/People's Daily/BIO"
mv example.dev "./data/People's Daily/BIO"
mv example.test "./data/People's Daily/BIO"
```

### Task

Named Entity Recognition

### Description

**Tags**: LOC(地名), ORG(机构名), PER(人名)  
**Tag Strategy**：BIO  
**Split**: '_space_' (北 B-LOC)  
**Data Size**:  
Train data set ( [example.train](example.train) ):

| 句数  | 字符数 | LOC 数 | ORG 数 | PER 数 |
| :---: | :----: | :----: | :----: | :----: |
| 20864 | 979180 | 16571  |  9277  |  8144  |

Dev data set ( [example.dev](example.dev) ):

| 句数 | 字符数 | LOC 数 | ORG 数 | PER 数 |
| :--: | :----: | :----: | :----: | :----: |
| 2318 | 109870 |  1951  |  984   |  884   |

Test data set ( [example.test](example.test) )

| 句数 | 字符数 | LOC 数 | ORG 数 | PER 数 |
| :--: | :----: | :----: | :----: | :----: |
| 4636 | 219197 |  3658  |  2185  |  1864  |

## References

- [1] https://github.com/OYE93/Chinese-NLP-Corpus
- [2] https://github.com/zjy-ucas/ChineseNER
