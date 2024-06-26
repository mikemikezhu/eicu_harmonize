# -*- coding: utf-8 -*-
"""export_mimic_drug_lab.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1YrsB4OzNCxzpKnDvOUgG-YZrhd2koaN9
"""

import pandas as pd
from datetime import datetime
import fasttext
import numpy as np
from tqdm import tqdm
import re

unique_drugs = ['acetamin', 'biotene', 'compazine', 'ferrous', 'imdur', 'lidocaine', 'milk of magnesia', 'nystatin', 'prochlorperazine', 'tamsulosin',
                'advair diskus', 'bisacodyl', 'coreg', 'flagyl', 'influenza vac', 'lipitor', 'mineral', 'omeprazole', 'promethazine', 'thiamine',
                'albumin', 'bumetanide', 'cozaar', 'flomax', 'infuvite', 'lisinopril', 'mineral oil', 'ondansetron', 'propofol', 'ticagrelor',
                'albuterol', 'bumex', 'decadron', 'flumazenil', 'insulin', 'lispro', 'mono-sod', 'optiray', 'pulmicort respule', 'tiotropium',
                'allopurinol', 'buminate', 'definity', 'fluticasone-salmeterol', 'insulin detemir', 'loratadine', 'morphine', 'oxycodone', 'quetiapine', 'toradol',
                'alprazolam', 'calcium carbonate', 'deltasone', 'folic acid', 'iohexol', 'lorazepam', 'motrin', 'pantoprazole', 'refresh p.m. op oint', 'tramadol',
                'alteplase', 'calcium chloride', 'dexamethasone', 'furosemide', 'iopamidol', 'losartan', 'mupirocin', 'parenteral nutrition', 'reglan', 'trandate',
                'alum hydroxide', 'calcium gluconate', 'dexmedetomidine', 'gabapentin', 'ipratropium', 'maalox', 'nafcillin', 'percocet', 'restoril', 'transde rm-scop',
                'ambien', 'cardizem', 'dextrose', 'glargine', 'isosorbide', 'magnesium chloride', 'naloxone', 'phenergan', 'ringers solution', 'trazodone',
                'aminocaproic acid', 'carvedilol', 'diazepam', 'glucagen', 'kayciel', 'magnesium hydroxide', 'narcan', 'phenylephrine', 'rocuronium', 'ultram',
                'amiodarone', 'catapres', 'digoxin', 'glucagon', 'kayexalate', 'magnesium oxide', 'neostigmine', 'phytonadione', 'roxicodone', 'valium',
                'amlodipine', 'cefazolin', 'diltiazem', 'glucose', 'keppra', 'magnesium sulf', 'neostigmine methylsulfate', 'piperacillin', 'sennosides', 'vancomycin',
                'anticoagulant', 'cefepime', 'diphenhydramine', 'glycopyrrolate', 'ketorolac', 'magox', 'neurontin', 'plasmalyte', 'seroquel', 'vasopressin',
                'apresoline', 'ceftriaxone', 'diprivan', 'guaifenesin', 'klonopin', 'medrol', 'nexterone', 'plavix', 'sertraline', 'ventolin',
                'ascorbic acid', 'cephulac', 'docusate', 'haldol', 'labetalol', 'meperidine', 'nicardipine', 'pneumococcal', 'simethicone', 'vitamin',
                'aspart', 'cetirizine', 'dopamine', 'haloperidol', 'lactated ringer', 'meropenem', 'nicoderm', 'pnu-immune-23', 'simvastatin', 'warfarin',
                'aspirin', 'chlorhexidine', 'ecotrin', 'heparin', 'lactulose', 'merrem', 'nicotine', 'polyethylene glycol', 'sodium bicarbonate', 'xanax',
                'atenolol', 'ciprofloxacin', 'enoxaparin', 'humulin', 'lanoxin', 'metformin', 'nitro-bid', 'potassium chloride', 'sodium chloride', 'zestril',
                'atorvastatin', 'cisatracurium', 'ephedrine', 'hydralazine', 'lantus', 'methylprednisolone', 'nitroglycerin', 'potassium phosphate', 'sodium phosphate', 'zocor',
                'atropine', 'citalopram', 'epinephrine', 'hydrochlorothiazide', 'levaquin', 'metoclopramide', 'nitroprusside', 'pravastatin', 'polystyrene sulfonate', 'zolpidem',
                'atrovent', 'clindamycin', 'etomidate', 'hydrocodone', 'levemir', 'metoprolol', 'norco', 'precedex', 'spironolactone', 'zosyn',
                'azithromycin', 'clonazepam', 'famotidine', 'hydrocortisone', 'levetiracetam', 'metronidazole', 'norepinephrine', 'prednisone', 'sublimaze',
                'bacitracin', 'clonidine', 'fat emulsion', 'hydromorphone', 'levofloxacin', 'midazolam', 'normodyne', 'prilocaine', 'succinylcholine',
                'bayer chewable', 'clopidogrel', 'fentanyl', 'ibuprofen', 'levothyroxine', 'midodrine', 'norvasc', 'prinivil', 'tacrolimus']


def get_age_group(age):

    if pd.isna(age):
        # 60 - 69 has the most patients
        return "60 - 69"

    if age == "> 89":
        return age

    age = int(age)

    if age < 30:
        return "< 30"
    elif age < 40:
        return "30 - 39"
    elif age < 50:
        return "40 - 49"
    elif age < 60:
        return "50 - 59"
    elif age < 70:
        return "60 - 69"
    elif age < 80:
        return "70 - 79"
    elif age < 90:
        return "80 - 89"
    else:
        return "> 89"


def get_race(race):

    if pd.isna(race) or "WHITE" in race:
        return "race_caucasion"
    elif "BLACK" in race:
        return "race_african"
    elif "HISPANIC" in race:
        return "race_hispanic"
    elif "ASIAN" in race:
        return "race_asian"
    else:
        return "race_caucasion"


# key: lab_name, value: lab_name in eicu
lab_name_mapping = {
    "o2sat": 50817, # OXYGEN SATURATION
    "pao2": 50821, # PO2
    "paco2": 50818, # PCO2
    "ph": 50820, # PH
    "albu_lab": 50862, # ALBUMIN
    "bands": 51144, # BANDS
    "bun": 51006, # UREA NITROGEN
    "hct": 51221, # HEMATOCRIT
    "inr": 51237, # INR(PT)
    "lactate": 50813, # LACTATE
    "platelets": 51265, # PLATELET COUNT
    "wbc": 51300 # WBC COUNT
}


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def calculate_similarity(model, drug1, drug2):
    drug1_vector = model.get_word_vector(drug1)
    drug2_vector = model.get_word_vector(drug2)
    return cosine_similarity(drug1_vector, drug2_vector)


def harmonize_drug(drug, unique_drugs, pretrained):

    if pd.isna(drug):
        return None

    drug = drug.lower()

    # Direct mapping: check whether drug names contain unique_drug
    for unique_drug in unique_drugs:
        if unique_drug in drug:
            return unique_drug

    # Check cosine similarity between word embeddings
    converted = re.sub(r'[^a-zA-Z\s]', '', drug)
    best_similarity = float('-inf')
    best_drug = None
    for drug_name in converted.split():
        for unique_drug in unique_drugs:
            similarity = calculate_similarity(
                pretrained, drug_name, unique_drug)
            if similarity > best_similarity:
                best_similarity = similarity
                best_drug = unique_drug

    return best_drug


def has_sepsis_diagnosis(text):
    has_sepsis = False
    text = text.lower()
    text = text.replace("\n", "")
    pattern = r"diagnosis:\s*(.+?):"
    matches = re.findall(pattern, text, re.DOTALL)
    for match in matches:
        if "sepsis" in match or "septic" in match:
            has_sepsis = True
    return has_sepsis


def calculate_diff_years(date1_str, date2_str):
    date1 = datetime.strptime(date1_str, "%Y-%m-%d %H:%M:%S")
    date2 = datetime.strptime(date2_str, "%Y-%m-%d %H:%M:%S")
    time_difference = date1 - date2
    return int(time_difference.days / 365.25)


def add_time_interval(start_str, interval_min):
    start_time = datetime.strptime(start_str, "%Y-%m-%d %H:%M:%S")
    cur_time = start_time + pd.Timedelta(minutes=interval_min)
    return cur_time.strftime("%Y-%m-%d %H:%M:%S")

# Admissions
mimic_admissions = pd.read_csv('data/mimic/ADMISSIONS.csv')
mimic_admissions = mimic_admissions[(mimic_admissions['ETHNICITY'] != 'OTHER') & (mimic_admissions['ETHNICITY'] != 'UNKNOWN/NOT SPECIFIED') & (mimic_admissions['ETHNICITY'] != 'UNABLE TO OBTAIN') & (mimic_admissions['ETHNICITY'] != 'PATIENT DECLINED TO ANSWER') & (mimic_admissions['ETHNICITY'] != 'MULTI RACE ETHNICITY')]
mimic_admissions.head()

# Patients
mimic_patients = pd.read_csv('data/mimic/PATIENTS.csv')
mimic_patients.head()

# Lab
mimic_lab = pd.read_csv('data/mimic/LABEVENTS.csv')
selected_mimic_lab = mimic_lab[mimic_lab['ITEMID'].isin(list(lab_name_mapping.values()))]
# filter out selected_mimic_lab whose valuenum is null
selected_mimic_lab = selected_mimic_lab[selected_mimic_lab['VALUENUM'].notnull()]

# Filter outliers
for mimic_lab_item in list(lab_name_mapping.values()):

    lab_values = selected_mimic_lab[selected_mimic_lab['ITEMID'] == mimic_lab_item]['VALUENUM']

    # Filter out outliers using IQR
    Q1 = lab_values.quantile(0.25)
    Q3 = lab_values.quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    lab_values_adjusted = lab_values.clip(lower_bound, upper_bound)

    mean_value = lab_values_adjusted.mean()
    std = lab_values_adjusted.std()

    # Normalize lab values using standard normal distribution
    lab_values_normalized = (lab_values_adjusted - mean_value) / std

    selected_mimic_lab.loc[selected_mimic_lab['ITEMID'] == mimic_lab_item, 'VALUENUM'] = lab_values_normalized

selected_mimic_lab = selected_mimic_lab[selected_mimic_lab['HADM_ID'].notnull()]
selected_mimic_lab.head()

# Drug
mimic_drug = pd.read_csv('data/mimic/PRESCRIPTIONS.csv')
mimic_drug = mimic_drug[mimic_drug['STARTDATE'].notnull()]
mimic_drug.head()

mimic_date_time_events = pd.read_csv('data/mimic/DATETIMEEVENTS.csv')
mimic_date_time_events.head()

# Drug harmonization
unique_drug_names = mimic_drug['DRUG'].unique()

pretrained = fasttext.load_model("data/pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")

harmonized_drug_dict = {drug: harmonize_drug(drug, unique_drugs, pretrained) for drug in unique_drug_names}
harmonized_drug_dict

# Ventilator
mimic_procedure = pd.read_csv('data/mimic/PROCEDUREEVENTS_MV.csv')
mimic_ventilator = mimic_procedure[mimic_procedure['ORDERCATEGORYNAME'].str.contains('Ventilation') | mimic_procedure['ORDERCATEGORYNAME'].str.contains('ventilation')]
mimic_ventilator.head()

# Sepsis
mimic_note = pd.read_csv('data/mimic/NOTEEVENTS.csv')
mimic_note = mimic_note[mimic_note['CHARTTIME'].notnull()]
mimic_sepsis = mimic_note[(mimic_note['TEXT'].str.contains('Sepsis') | mimic_note['TEXT'].str.contains('sepsis')) & (mimic_note['TEXT'].str.contains('Diagnosis:') | mimic_note['TEXT'].str.contains('diagnosis:'))]
mimic_sepsis = mimic_sepsis[mimic_sepsis['TEXT'].apply(has_sepsis_diagnosis)]
mimic_sepsis.head()

# Select patients whose have all records
admission_patient = mimic_admissions['SUBJECT_ID'].unique()
lab_patient = selected_mimic_lab['SUBJECT_ID'].unique()
drug_patient = mimic_drug['SUBJECT_ID'].unique()
procedure_patient = mimic_procedure['SUBJECT_ID'].unique()
note_patient = mimic_note['SUBJECT_ID'].unique()
record_patients = mimic_patients['SUBJECT_ID'].unique()
selected_patients = list(set(admission_patient) & set(lab_patient) & set(drug_patient) & set(procedure_patient) & set(note_patient) & set(record_patients))
len(selected_patients)

max(selected_patients), min(selected_patients)

# Select patients whose value is from 0 to 30000
selected_patients = [patient for patient in selected_patients if patient > 60000 and patient <= 70000]
len(selected_patients)

# MIMIC columns should align with eICU columns
eicu_lab_data = pd.read_csv('eicu_drug_lab.csv')
eicu_columns = eicu_lab_data.columns
mimic_data = pd.DataFrame(columns=eicu_columns)
mimic_data.head()

observe_time = 48 * 60 # 48 hours

for index, patient in tqdm(enumerate(selected_patients)):

    mimic_row = pd.DataFrame(0.0, index=[0], columns=eicu_columns)

    # Patient id
    mimic_row['patientunitstayid'] = patient

    # Hospital id
    mimic_row['hospitalid'] = -1.0 # MIMIC hospitals were set to -1.0

    # Admission: If there are multiple admissions, choose the one with the most recent admission
    admission_data = mimic_admissions[mimic_admissions['SUBJECT_ID'] == patient]
    admission_data = admission_data.sort_values(by='ADMITTIME', ascending=False).head(1)
    hadm_id = admission_data['HADM_ID'].values[0]
    admit_time = admission_data['ADMITTIME'].values[0]

    # Check death after 48 hours of admission
    admit_time_48hrs = add_time_interval(admit_time, observe_time)

    # Age
    patient_data = mimic_patients[mimic_patients['SUBJECT_ID'] == patient]
    dob = patient_data['DOB'].values[0]
    age = calculate_diff_years(admit_time, dob) # Admission time - date of birth
    age_group = get_age_group(age)
    mimic_row[age_group] = 1.0

    # Sex
    sex_value = patient_data['GENDER'].values[0]
    sex = "sex_is_female" if sex_value == 'F' else "sex_is_male"
    mimic_row[sex] = 1.0

    # Race
    race_value = admission_data['ETHNICITY'].values[0]
    race = get_race(race_value)
    mimic_row[race] = 1.0

    # BMI: BMI needs CHARTEVENTS table that we don't have it. So we will use normal BMI
    mimic_row["bmi_normal"] = 1.0

    # Death
    selected_death = admission_data[(admission_data['HOSPITAL_EXPIRE_FLAG'] == 1.0) & (admission_data['DEATHTIME'] >= admit_time_48hrs)]
    patient_death = 1.0 if selected_death.shape[0] > 0 else 0.0
    mimic_row['death'] = patient_death

    # Drug
    # Because MIMIC-III doesn't provide the exact start time of medications, and only provides approximate dates. We will check whether the medication was prescribed during the observation window (prescribed start time <= observation window end time and prescribed end time >= observation window start time).
    drugs = mimic_drug[(mimic_drug['SUBJECT_ID'] == patient) & (mimic_drug['HADM_ID'] == hadm_id) & (mimic_drug['STARTDATE'] <= admit_time_48hrs)]['DRUG'].to_list()
    for drug in drugs:
        converted_drug = harmonized_drug_dict.get(drug)
        if converted_drug is not None:
            mimic_row[converted_drug] = 1.0

    # Lab
    mimic_lab_patient = selected_mimic_lab[(selected_mimic_lab['SUBJECT_ID'] == patient) & (selected_mimic_lab['HADM_ID'] == hadm_id) & (selected_mimic_lab['CHARTTIME'] >= admit_time) & (selected_mimic_lab['CHARTTIME'] < admit_time_48hrs)]

    for lab_name, mimic_lab_item in lab_name_mapping.items():
        lab_value = mimic_lab_patient[mimic_lab_patient['ITEMID'] == mimic_lab_item]['VALUENUM'].values

        # average lab values
        if lab_value.shape[0] > 0:
            mimic_row[lab_name] = lab_value.mean()

    mimic_data = pd.concat([mimic_data, mimic_row], ignore_index=True)

mimic_data.to_csv('mimic_drug_lab_4.csv', index=False)

mimic_data.head()