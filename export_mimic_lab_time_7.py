# -*- coding: utf-8 -*-
"""export_mimic_lab_time.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16r9zJFFKoCd8tCYtvNC2wjCpowerK2Lj
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

# Test has_sepsis_diagnosis
# text = "Mr [**Known lastname 778**] is a 79 yr. gentleman, with hx of Alzheimers dementia,\n   HTN, CHF and AF. On [**2-5**] pt adm to osh with fever and chills . pt. noted\n   to have a lll influenza pneumonia, and sepsis and required intubation\n   on [**2-5**] and transferred to the micu/.sicu for further management.\n   Pt. did have an episode of hypotension requiring pressors, but just for\n   a short time. Pt. extubated on [**2-10**] , and respitory wise did well with\n   sats in high 90\ns on room air.\n   Pt. had been on versed during his intubation and on [**2-12**] pt. noted to\n   have decreased responsiveness with no verbal response. Drooling from\n   left side of mouth, interm.,jerking movements. Neuro consulted- Dx\n   with myoclonus movements which could be attributed to versed or his\n   alzheimers.\n   Pneumonia, viral\n   Assessment:\n   Gradual daily improvement with broadening of antibx [**2-14**] measured by\n   SpO2>95% on R/A, Afebrile\n   Action:\n   CXR: [**2-16**] persistent LL collapse with small-mod bilateral pleural\n   effusions\n   Cont. on antibiotics.\n   O2 D/C\ned; NT suctioning prn\n   Response:\n   Stable on R/A\n   Plan:\n   Continue to monitor s&sx of infection; resent sputum Cx if possible,\n   Cont on antibiotics, DB&C-NT sx prn, Increase activity as tolerated\n   Altered mental status (not Delirium)\n   Assessment:\n   Remains Delirious improving daily as infection clearing\n   Action:\n   Observation only. No tx at this time. EEG was done on [**2-14**] which was\n   neg. LP and MRI not indicated at this time\n   Hand mitt restraints bilaterally to protect IV and NGT\n   Safety measures-bed in low locked position, close observation, bed\n   alarm activated\n   Remains off all sedatives\n   Response:\n   More responsive today now that he is afebrile and back on broad\n   spectrum antibx , follows commands intermittently, speech is garbled\n   but understandable, appears closer to baseline as per wife\n[**Initials (NamePattern4) **] [**Last Name (NamePattern4) 1042**]\n   Plan:\n   Continue to monitor MS, neurology consulting, continue to minimize all\n   sedatives, safety measures, sitter recommended\n   Atrial fibrillation (Afib)\n   Assessment:\n   Continues in AF with occass pvc\n   Action:\n   Cardiac meds adjusted to optimize rate control\n   Lopressor, Diltiazem (started today) and Captorpril (low-dose)\n   Electrolytes repleated today (K and neutrophos)\n   Response:\n   Hemodynamically stable with better rate control (HR:95-110)\n   Plan:\n   Continue to monitor HR, adjust meds prn-Dilt seems to be most effective\n   at rate control\n   Retroperitoneal bleed (RP bleed), spontaneous\n   Assessment:\n   Ct + for bleed. Left flank ecchymotic. No c/o pain-site unchanged\n   Action:\n   Heparin held; 1u pc\ns given\n   Response:\n   Stable hct 31-32 Off heparin without evidence of further bleeding\n   Plan:\n   Monitor S&Sx of bleeding, Hct check with am labs, SC Heparin started\n   today\n   Demographics\n   Attending MD:\n   [**Location (un) 832**]\n   Admit diagnosis:\n   SEPSIS\n   Code status:\n   Full code\n   Height:\n   68 Inch\n   Admission weight:\n   67 kg\n   Daily weight:\n   67.5 kg\n   Allergies/Reactions:\n   No Known Drug Allergies\n   Precautions: Standard\n   PMH:\n   CV-PMH:  CHF, Hypertension\n   Additional history: alzteimers, dementia, Afib\n   Surgery / Procedure and date:\n   Latest Vital Signs and I/O\n   Non-invasive BP:\n   S:110\n   D:27\n   Temperature:\n   98.4\n   Arterial BP:\n   S:125\n   D:73\n   Respiratory rate:\n   25 insp/min\n   Heart Rate:\n   105 bpm\n   Heart rhythm:\n   AF (Atrial Fibrillation)\n   O2 delivery device:\n   None\n   O2 saturation:\n   95% %\n   O2 flow:\n   FiO2 set:\n   50% %\n   24h total in:\n   1,787 mL\n   24h total out:\n   1,610 mL\n   Pertinent Lab Results:\n   Sodium:\n   139 mEq/L\n   [**2151-2-17**] 05:13 AM\n   Potassium:\n   3.5 mEq/L\n   [**2151-2-17**] 05:13 AM\n   Chloride:\n   107 mEq/L\n   [**2151-2-17**] 05:13 AM\n   CO2:\n   24 mEq/L\n   [**2151-2-17**] 05:13 AM\n   BUN:\n   19 mg/dL\n   [**2151-2-17**] 05:13 AM\n   Creatinine:\n   0.8 mg/dL\n   [**2151-2-17**] 05:13 AM\n   Glucose:\n   126 mg/dL\n   [**2151-2-17**] 05:13 AM\n   Hematocrit:\n   31.1 %\n   [**2151-2-17**] 05:13 AM\n   Finger Stick Glucose:\n   125\n   [**2151-2-17**] 12:00 PM\n   Additional pertinent labs:\n   Lines / Tubes / Drains:\n   Indwelling Foley catheter, [**Hospital1 974**] Sump NGT R nare\n   Valuables / Signature\n   Patient valuables: None\n   Clothes:  None\n   Wallet / Money:\n   No money / wallet\n   Jewelry: Gold wedding [**Hospital 1044**]\n   Transferred from: [**Hospital 44**]\n   Transferred to: 11 [**Hospital Ward Name 89**]\n   Date & time of Transfer: [**2151-2-17**]\n"
# has_sepsis_diagnosis(text)

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
selected_patients = [patient for patient in selected_patients if patient > 90000]
len(selected_patients)

# MIMIC columns should align with eICU columns
eicu_lab_data = pd.read_csv('eicu_lab_time.csv')
eicu_columns = eicu_lab_data.columns
mimic_data = pd.DataFrame(columns=eicu_columns)
mimic_data.head()

observation_time = (72 + 12) * 60 # 72 hours (observation) + 12 hours (pre-observation)
bin_time = 12 * 60
bin = observation_time // bin_time

for index, patient in tqdm(enumerate(selected_patients)):

    for cur_bin in range(bin):

        mimic_row = pd.DataFrame(0.0, index=[0], columns=eicu_columns)

        # Patient id
        mimic_row['patientunitstayid'] = patient

        # Hospital id
        mimic_row['hospitalid'] = -1.0 # MIMIC hospitals were set to -1.0

        # Time window: Start from -1 to 5
        observation_bin = cur_bin - 1
        mimic_row['time_window'] = observation_bin

        # Admission: If there are multiple admissions, choose the one with the most recent admission
        admission_data = mimic_admissions[mimic_admissions['SUBJECT_ID'] == patient]
        admission_data = admission_data.sort_values(by='ADMITTIME', ascending=False).head(1)
        hadm_id = admission_data['HADM_ID'].values[0]
        admit_time = admission_data['ADMITTIME'].values[0]

        # Observing start: admit_time + cur_bin * bin_time
        start_time = add_time_interval(admit_time, cur_bin * bin_time)
        end_time = add_time_interval(admit_time, (cur_bin + 1) * bin_time)

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
        selected_death = admission_data[(admission_data['HOSPITAL_EXPIRE_FLAG'] == 1.0) & (admission_data['DEATHTIME'] >= start_time) & (admission_data['DEATHTIME'] < end_time)]
        patient_death = 1.0 if selected_death.shape[0] > 0 else 0.0
        mimic_row['death'] = patient_death

        # Ventilator
        selected_ventilator = mimic_ventilator[(mimic_ventilator['SUBJECT_ID'] == patient) & (mimic_ventilator['HADM_ID'] == hadm_id) & (mimic_ventilator['STARTTIME'] >= start_time) & (mimic_ventilator['STARTTIME'] < end_time)]
        patient_ventilator = 1.0 if selected_ventilator.shape[0] > 0 else 0.0
        mimic_row['ventilator'] = patient_ventilator

        # Sepsis
        selected_sepsis = mimic_sepsis[(mimic_sepsis['SUBJECT_ID'] == patient) & (mimic_sepsis['HADM_ID'] == hadm_id)  & (mimic_sepsis['CHARTTIME'] >= start_time) & (mimic_sepsis['CHARTTIME'] < end_time)]
        patient_sepsis = 1.0 if selected_sepsis.shape[0] > 0 else 0.0
        mimic_row['sepsis'] = patient_sepsis

        # Drug
        # Because MIMIC-III doesn't provide the exact start time of medications, and only provides approximate dates. We will check whether the medication was prescribed during the observation window (prescribed start time <= observation window end time and prescribed end time >= observation window start time).
        drugs = mimic_drug[(mimic_drug['SUBJECT_ID'] == patient) & (mimic_drug['HADM_ID'] == hadm_id) & (mimic_drug['STARTDATE'] <= end_time) & (mimic_drug['ENDDATE'] >= add_time_interval(start_time, -24*60))]['DRUG'].to_list()
        for drug in drugs:
            converted_drug = harmonized_drug_dict.get(drug)
            if converted_drug is not None:
                mimic_row[converted_drug] = 1.0

        # Lab
        mimic_lab_patient = selected_mimic_lab[(selected_mimic_lab['SUBJECT_ID'] == patient) & (selected_mimic_lab['HADM_ID'] == hadm_id) & (selected_mimic_lab['CHARTTIME'] >= start_time) & (selected_mimic_lab['CHARTTIME'] < end_time)]

        for lab_name, mimic_lab_item in lab_name_mapping.items():
            lab_value = mimic_lab_patient[mimic_lab_patient['ITEMID'] == mimic_lab_item]['VALUENUM'].values

            # average lab values
            if lab_value.shape[0] > 0:
                mimic_row[lab_name] = lab_value.mean()

        mimic_data = pd.concat([mimic_data, mimic_row], ignore_index=True)

mimic_data.to_csv('mimic_lab_time_7.csv', index=False)