# Data Harmonization for eICU Collaborative Research Database

## Introduction

TODO

## Impute Missing Drug Names

TODO

## Drug Name Harmonization

TODO

## User Manual
- Create `data` directory under the root file
    - Inside `data` directory, create `eicu` and `pretrained` subdirectory
    - Download the eICU Collaborative Research Database and put it inside the `data/eicu` folder
    - Download the BioWordVec model from https://github.com/ncbi-nlp/BioSentVec and put it inside the `data/pretrained` folder
- Create `output` directory under the root file
- Run `main.py` script
- Find the harmonized dataset `eicu_harmonized.csv` inside `output` folder