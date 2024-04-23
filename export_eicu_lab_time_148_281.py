import pandas as pd
from tqdm import tqdm

eicu_time = pd.read_csv("eicu_time_dataset.csv")
lab = pd.read_csv("data/eicu/lab.csv")

# filter eicu_time by hospital 148 or hospital 281
eicu_time = eicu_time[(eicu_time['hospitalid'] == 148) | (eicu_time['hospitalid'] == 281)]

# key: lab_name, value: lab_name in eicu
lab_name_mapping = {
    "o2sat": "O2 Sat (%)",
    "pao2": "paO2",
    "paco2": "paCO2",
    "ph": "pH",
    "albu_lab": "albumin",
    "bands": "-bands",
    "bun": "BUN",
    "hct": "Hct",
    "inr": "PT - INR",
    "lactate": "lactate",
    "platelets": "platelets x 1000",
    "wbc": "WBC x 1000"
}

lab_name_mapping_values = list(lab_name_mapping.values())

selected_lab = lab[lab['labname'].isin(lab_name_mapping_values)].copy()

for value in lab_name_mapping_values:

    lab_values = selected_lab[selected_lab['labname'] == value]['labresult']

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

    selected_lab.loc[selected_lab['labname'] == value, 'labresult'] = lab_values_normalized


eicu_columns = list(eicu_time.columns) + list(lab_name_mapping.keys())
eicu_lab_time = pd.DataFrame(columns=eicu_columns)

bin_time = 12 * 60

for index, row in tqdm(eicu_time.iterrows(), total=eicu_time.shape[0]):

    eicu_row = pd.DataFrame(0.0, index=[0], columns=eicu_columns)

    # copy row values to eicu_row
    for column in eicu_time.columns:
        eicu_row[column] = row[column]

    # get patient lab data
    patient_id = row['patientunitstayid']
    lab_patient = selected_lab[selected_lab['patientunitstayid'] == patient_id]

    time_window = row['time_window']
    if time_window == -1:
        time_window_start = float('-inf')
        time_window_end = 0
    else:
        time_window_start = time_window * bin_time
        time_window_end = (time_window + 1) * bin_time

    lab_time_patient = lab_patient[(lab_patient['labresultoffset'] >= time_window_start) & (lab_patient['labresultoffset'] < time_window_end)]

    for lab_name, eicu_lab_name in lab_name_mapping.items():
        lab_value = lab_time_patient[lab_time_patient['labname'] == eicu_lab_name]['labresult']

        # average lab values
        if lab_value.shape[0] > 0:
            eicu_row[lab_name] = lab_value.mean()

    eicu_lab_time = pd.concat([eicu_lab_time, eicu_row], ignore_index=True)

eicu_lab_time.to_csv('eicu_lab_time_148_281.csv', index=False)