import pandas as pd
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


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def calculate_similarity(model, drug1, drug2):
    drug1_vector = model.get_word_vector(drug1)
    drug2_vector = model.get_word_vector(drug2)
    return cosine_similarity(drug1_vector, drug2_vector)


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


def get_bmi(weight_kg, height_cm):

    if pd.isna(weight_kg) or pd.isna(height_cm) or weight_kg == 0 or height_cm == 0:
        return "bmi_normal"

    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)

    if bmi < 18.5:
        return "bmi_underweight"
    elif bmi < 24.9:
        return "bmi_normal"
    elif bmi < 29.9:
        return "bmi_overweight"
    else:
        return "bmi_obesity"


def get_race(race):

    if pd.isna(race) or race == "Caucasian":
        return "race_caucasion"
    elif race == "African American":
        return "race_african"
    elif race == "Hispanic":
        return "race_hispanic"
    elif race == "Asian":
        return "race_asian"
    elif race == "Native American":
        return "race_native"
    else:
        return "race_caucasion"


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

import pandas as pd

patient = pd.read_csv("data/eicu/patient.csv")
medication_imputed = pd.read_csv("output/medication_imputed.csv")
diagnosis = pd.read_csv("data/eicu/diagnosis.csv")
treatment = pd.read_csv("data/eicu/treatment.csv")
patient_medication = patient.merge(medication_imputed, on='patientunitstayid', how='inner')
# patient_medication = patient_medication[patient_medication["hospitalid"].isin([167,420,199,458,252,165,148,281,449,283])]
patient_medication = patient_medication[patient_medication["hospitalid"].isin([167,420])]
ventilator_treatments = treatment[treatment['treatmentstring'].str.contains('ventilation', case=False, na=False)]

sepsis_diagnosis = diagnosis[diagnosis['diagnosisstring'].str.contains('sepsis', case=False, na=False)]

mortaility_patient = patient[patient["unitdischargestatus"] == 'Expired']

eicu_time = pd.DataFrame(columns=['patientunitstayid', 'hospitalid', 'time_window', 'ventilator', 'sepsis'] + unique_drugs + ['bmi_underweight', 'bmi_normal', 'bmi_overweight', 'bmi_obesity', 'race_african', 'race_hispanic', 'race_caucasion', 'race_asian', 'race_native', 'sex_is_male', 'sex_is_female', '< 30', '30 - 39', '40 - 49', '50 - 59', '60 - 69', '70 - 79', '80 - 89', '> 89'])

observation_time = 72 * 60
bin_time = 12 * 60
bin = observation_time // bin_time

pretrained = fasttext.load_model("data/pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")

unique_drug_names = patient_medication['drugname'].unique()
harmonized_drug_dict = {drug: harmonize_drug(drug, unique_drugs, pretrained) for drug in unique_drug_names}

patient = patient[patient["hospitalid"].isin([167,420])]
for index, row in tqdm(patient.iterrows(), total=patient.shape[0]):

    pre_eicu_row = pd.DataFrame(0.0, index=[0], columns=eicu_time.columns)

    patientunitstayid = row['patientunitstayid']

    pre_eicu_row['patientunitstayid'] = float(patientunitstayid)
    pre_eicu_row['hospitalid'] = float(row['hospitalid'])
    pre_eicu_row['time_window'] = -1.0

    # BMI
    bmi = get_bmi(row['admissionweight'], row['admissionheight'])
    pre_eicu_row[bmi] = 1.0

    # Race
    race = get_race(row['ethnicity'])
    pre_eicu_row[race] = 1.0

    # Sex
    sex = "sex_is_female" if row['gender'] == 'Female' else "sex_is_male"
    pre_eicu_row[sex] = 1.0

    # Age
    age_group = get_age_group(row['age'])
    pre_eicu_row[age_group] = 1.0

    # Drug
    time_window_drug = patient_medication[(patient_medication["patientunitstayid"] == patientunitstayid) & (
            patient_medication["drugstartoffset"] < 0)]

    for drug in time_window_drug["drugname"]:

        converted_drug = harmonized_drug_dict.get(drug)
        if converted_drug is not None:
            pre_eicu_row[converted_drug] = 1.0

    eicu_time = pd.concat([eicu_time, pre_eicu_row], ignore_index=True)

    for cur_bin in range(bin):
            
        eicu_row = pd.DataFrame(0.0, index=[0], columns=eicu_time.columns)
    
        patientunitstayid = row['patientunitstayid']
        
        eicu_row['patientunitstayid'] = float(patientunitstayid)
        eicu_row['hospitalid'] = float(row['hospitalid'])
        eicu_row['time_window'] = float(cur_bin)
        
        # BMI
        bmi = get_bmi(row['admissionweight'], row['admissionheight'])
        eicu_row[bmi] = 1.0
    
        # Race
        race = get_race(row['ethnicity'])
        eicu_row[race] = 1.0
    
        # Sex
        sex = "sex_is_female" if row['gender'] == 'Female' else "sex_is_male"
        eicu_row[sex] = 1.0
    
        # Age
        age_group = get_age_group(row['age'])
        eicu_row[age_group] = 1.0

        time_window_start = cur_bin * bin_time
        time_window_end = (cur_bin + 1) * bin_time

        # Label
        eicu_row["ventilator"] = float(len(ventilator_treatments[
                                               (ventilator_treatments["treatmentoffset"] >= time_window_start) & (
                                                           ventilator_treatments[
                                                               "treatmentoffset"] < time_window_end) & (
                                                           ventilator_treatments[
                                                               "patientunitstayid"] == patientunitstayid)]) > 0)

        eicu_row["sepsis"] = float(len(sepsis_diagnosis[(sepsis_diagnosis["diagnosisoffset"] >= time_window_start) & (
                    sepsis_diagnosis["diagnosisoffset"] < time_window_end) & (sepsis_diagnosis[
                                                                                  "patientunitstayid"] == patientunitstayid)]) > 0)
        
        eicu_row["death"] = float(len(mortaility_patient[(mortaility_patient["unitdischargeoffset"] >= time_window_start) & (
                    mortaility_patient["unitdischargeoffset"] < time_window_end) & (mortaility_patient["patientunitstayid"] == patientunitstayid)]) > 0)

        # Drug
        time_window_drug = patient_medication[(patient_medication["patientunitstayid"] == patientunitstayid) & (patient_medication["drugstartoffset"] >= time_window_start) & (patient_medication["drugstartoffset"] < time_window_end)]

        for drug in time_window_drug["drugname"]:
            
            converted_drug = harmonized_drug_dict.get(drug)
            if converted_drug is not None:
                eicu_row[converted_drug] = 1.0
        
        eicu_time = pd.concat([eicu_time, eicu_row], ignore_index=True)

print("Export time series eicu dataset...")
eicu_time.to_csv('output/eicu_time_167_420.csv', index=False)
eicu_time.head()
        