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


admissions = pd.read_csv('data/mimic/ADMISSIONS.csv')
admissions = admissions[admissions['SUBJECT_ID'] < 20000]
mimic_lab = pd.read_csv('data/mimic/LABEVENTS.csv')
selected_mimic_lab = mimic_lab[mimic_lab['ITEMID'].isin(list(lab_name_mapping.values()))]
selected_mimic_lab = selected_mimic_lab[selected_mimic_lab['VALUENUM'].notnull()]

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


mimic_drug = pd.read_csv('data/mimic/PRESCRIPTIONS.csv')

admission_patient = admissions['SUBJECT_ID'].unique()
lab_patient = selected_mimic_lab['SUBJECT_ID'].unique()
drug_patient = mimic_drug['SUBJECT_ID'].unique()

unique_drug_names = mimic_drug['DRUG'].unique()

pretrained = fasttext.load_model("data/pretrained/BioWordVec_PubMed_MIMICIII_d200.bin")

harmonized_drug_dict = {drug: harmonize_drug(drug, unique_drugs, pretrained) for drug in unique_drug_names}


eicu_lab_data = pd.read_csv('eicu_lab_time.csv')
eicu_columns = list(eicu_lab_data.columns[1:3]) + list(eicu_lab_data.columns[4:5]) + list(eicu_lab_data.columns[7:-31]) + list(eicu_lab_data.columns[-12:])

mimic_data = pd.DataFrame(columns=eicu_columns)

# Find intersection
selected_patients = set(admission_patient) & set(lab_patient) & set(drug_patient)

for index, patient in tqdm(enumerate(selected_patients)):

    mimic_row = pd.DataFrame(0.0, index=[0], columns=eicu_columns)

    # Patient id
    mimic_row['patientunitstayid'] = patient

    # Hospital id
    mimic_row['hospitalid'] = -1.0 # MIMIC

    # Death
    patient_death = admissions[admissions['SUBJECT_ID'] == patient]['HOSPITAL_EXPIRE_FLAG'].values[0]
    mimic_row['death'] = patient_death

    # Drug
    drugs = mimic_drug[mimic_drug['SUBJECT_ID'] == patient]['DRUG'].to_list()
    for drug in drugs:
        converted_drug = harmonized_drug_dict.get(drug)
        if converted_drug is not None:
            mimic_row[converted_drug] = 1.0

    # Lab
    mimic_lab_patient = selected_mimic_lab[selected_mimic_lab['SUBJECT_ID'] == patient]

    for lab_name, mimic_lab_item in lab_name_mapping.items():
        lab_value = mimic_lab_patient[mimic_lab_patient['ITEMID'] == mimic_lab_item]['VALUENUM'].values

        # average lab values
        if lab_value.shape[0] > 0:
            print(lab_name, lab_value.mean())
            mimic_row[lab_name] = lab_value.mean()

    mimic_data = pd.concat([mimic_data, mimic_row], ignore_index=True)


mimic_data.to_csv('mimic_data_1.csv', index=False)