# Data Harmonization for eICU Collaborative Research Database

## Introduction

The eICU Collaborative Research Database is pivotal for understanding intensive care unit patient data. However, there are three significant challenges when training models using the eICU dataset.
- Approximately one-third of drug names remain unrecorded.
- Some drugs share the same identity (e.g. "aspirin" and "acetylsalicylic acid"), and patient data can be merged across all identities of said medications. 
- The dosage information in the drug names can be disregarded (e.g. "aspirin 10 mg").

This project aims to combine drug names with the same identity and harmonize the eICU dataset, making it easier for researchers to train models effectively. The harmonized eICU dataset can be downloaded at https://drive.google.com/file/d/1tpvntkx4nCUISEvHvCOOw4UgLELNuM9b/view?usp=sharing.

## Impute Missing Drug Names

- First, extract records from the `medication.csv` of eICU dataset where both drug names and their respective HICL codes are available.

- Then, construct a dictionary mapping between drug HICL codes and their corresponding drug names.

- For medication records in the eICU dataset with missing drug names but available drug HICL codes, impute the missing drug names using the prepared dictionary.

The imputed eICU medication data is exported to `medication_imputed.csv` inside `output` folder for future analyses.

The percentage distribution of drug existance across hospitals before / after imputation is as follows:

![](img/drugname_imputation.png)

## Drug Name Harmonization

There are two steps to harmonize drug names:

1. Initial Drug Mapping Check

    - Prepare a reference drug names with 237 most common drugs
    - Assume we have a drug called "A 10mg"
	- Loop through all the drug names in the reference drug names:
	    - Check if the reference drug name (e.g., "A") is a part of the given drug name ("A 10mg")
	    - If a match is found, directly map the given drug name to the reference drug name (e.g., “A")

2. Use BioWordVec for Mapping

    - If a direct match isn't found, start the similarity process:
        - Loop through each reference drug name
        - Convert both the given drug name and the current reference drug name to their respective word embeddings using BioWordVec
        - Compute the cosine similarity between the two word embeddings
        - After looping through all reference drug names, choose the reference drug name with the largest cosine similarity

The overlap percentage before / after harmonization is as follows:

![](img/drugname_harmonization.png)

## Create Harmonized eICU Dataset

The dataset includes 200,859 patients treated in 208 hospitals. Each patient's information comprises several features, including the drugs, age, sex, race, and BMI. These characteristics are represented in a binary format. Specifically, when a feature is represented by the value 1, it indicates that the patients belong to a specific category. For example, if a patient's age, gender, race, or BMI feature is marked as 1, they fall within a particular age group, gender category, race group, or BMI range.
Similarly, when a drug feature is indicated as 1, the patient has been prescribed that specific drug. The dataset also encompasses three outcome labels: death, ventilator use, and sepsis. These labels are also binary, with a value of 1 indicating the presence of a specific outcome or condition for the patient, such as death, ventilator support, or sepsis.

## User Manual
- Create `data` directory under the root file
    - Inside `data` directory, create `eicu` and `pretrained` subdirectory
    - Download the eICU Collaborative Research Database and put it inside the `data/eicu` folder
    - Download the BioWordVec model from https://github.com/ncbi-nlp/BioSentVec and put it inside the `data/pretrained` folder
- Create `output` directory under the root file
- Run `main.py` script. It may take a couple of hours.
- Find the harmonized dataset `eicu_harmonized.csv` inside `output` folder

## Citation

```
TODO
```