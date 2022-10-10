## MSRA dataset

### Data Sources

Download **original** MSRA data from: https://github.com/OYE93/Chinese-NLP-Corpus/tree/master/NER/MSRA, then:

```bash
mkdir -p ./data/MSRA/BIO
mv msra_train_bio ./data/MSRA/BIO/train.tsv
mv msra_test_bio ./data/MSRA/BIO/test.tsv
```

### Task

Named Entity Recognition

### Description

**Tags**: LOC(地名), ORG(机构名), PER(人名)  
**Tag Strategy**：BIO  
**Split**: '\t' (北\tB-LOC)  
**Data Size**:  
Train data set ( [msra_train_bio.txt](msra_train_bio.txt) ):

| 句数  | 字符数  | LOC 数 | ORG 数 | PER 数 |
| :---: | :-----: | :----: | :----: | :----: |
| 45000 | 2171573 | 36860  | 20584  | 17615  |

Test data set ( [msra_test_bio.txt](msra_test_bio.txt) )

| 句数 | 字符数 | LOC 数 | ORG 数 | PER 数 |
| :--: | :----: | :----: | :----: | :----: |
| 3442 | 172601 |  2886  |  1331  |  1973  |

## References

- [1] https://github.com/OYE93/Chinese-NLP-Corpus
- [2] [The third international Chinese language processing bakeoff: Word segmentation and named entity recognition](https://faculty.washington.edu/levow/papers/sighan06.pdf)
- [3] https://github.com/bytetopia/nlp_datasets
