# CinPatent: Datasets for Patent Classification

We release two datasets for patent classification in English and Japanese at [Google Drive](https://drive.google.com/drive/folders/1Y5pIVqM1D-Bl_MerQidLxVf2mU4Iapas?usp=sharing). The data folder structure upon extracted is described in section [Structure](#structure).

## Data description
Each data file is a `.ndjson` in which each line describes a sample in 
json format with following attributes.
| Field | Data type | Meaning |
| --- | --- | --- |
| id | string | Patent ID |
| title | string | Patent title |
| abstract | string | Patent abstract |
| claim_1 | string | First claim from patent claims |
| claims | string | All patent claims |
| description | string | Patent description |
| is_train | boolean | Whether the sample is for training |
| is_dev | boolean | Whether the sample is for development |
| is_test | boolean | Whether the sample is for testing |

We partition data with ratio 80:10:10 for training, development, and testing. Following table provides several statistics of our datasets.
| | CinPatent-EN | CinPatent-JA |
| --- | --- | --- |
| no. samples | 45,131 | 54,657 |
| no. labels | 425 | 523 |
| no. samples/label | 221.69 ± 38.56 | 226.94 ± 41.74 |
| no. labels/sample | 2.09 ± 1.31 | 2.17 ± 1.32 |

## Structure

The datasets are available with multiple ratios: 10%, 25%, 50%, 75%, and 100%.

```
data
├── CinPatent-EN
│   ├── en_0.05.ndjson
│   ├── en_0.1.ndjson
│   ├── en_0.25.ndjson
│   ├── en_0.5.ndjson
│   ├── en_0.75.ndjson
│   └── en_1.0.ndjson
└── CinPatent-JA
    ├── ja_0.05.ndjson
    ├── ja_0.1.ndjson
    ├── ja_0.25.ndjson
    ├── ja_0.5.ndjson
    ├── ja_0.75.ndjson
    └── ja_1.0.ndjson
```

## Contact
For further support, please contact us at [joanna@cinnamon.is](joanna@cinnamon.is).
