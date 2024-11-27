# generate_hospital_readmissions.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_hospital_data(n_patients=10000, seed=42):
    """
    Generate synthetic hospital readmission dataset.

    Parameters:
    -----------
    n_patients : int
        Number of patient records to generate.
    seed : int
        Random seed for reproducibility.

    Returns:
    --------
    pandas.DataFrame
        Synthetic hospital readmission dataset.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Generate patient IDs
    patient_ids = range(1, n_patients + 1)

    # Generate ages with a realistic distribution (mostly adults and elderly)
    ages = np.random.normal(65, 15, n_patients)
    ages = np.clip(ages, 18, 100).astype(int)

    # Generate gender
    genders = np.random.choice(['M', 'F'], size=n_patients, p=[0.48, 0.52])

    # Generate admission types with realistic probabilities
    admission_types = np.random.choice(
        ['Emergency', 'Elective', 'Urgent'],
        size=n_patients,
        p=[0.6, 0.3, 0.1]
    )

    # Common primary diagnoses with ICD-10 codes
    diagnoses = [
        ('I25.10', 'Coronary artery disease'),
        ('I50.9', 'Heart failure'),
        ('J44.9', 'COPD'),
        ('E11.9', 'Type 2 diabetes'),
        ('J18.9', 'Pneumonia'),
        ('N17.9', 'Acute kidney failure'),
        ('K92.2', 'Gastrointestinal bleeding'),
        ('I63.9', 'Cerebral infarction'),
        ('I48.91', 'Atrial fibrillation'),
        ('F32.9', 'Major depressive disorder')
    ]

    # Generate diagnoses with realistic probabilities
    diagnosis_probs = [0.15, 0.15, 0.12, 0.12, 0.1, 0.1, 0.08, 0.08, 0.05, 0.05]
    primary_diagnoses = np.random.choice(
        [d[0] for d in diagnoses],
        size=n_patients,
        p=diagnosis_probs
    )

    # Generate diagnosis descriptions
    diagnosis_dict = dict(diagnoses)
    diagnosis_descriptions = [diagnosis_dict[code] for code in primary_diagnoses]

    # Generate length of stay (LOS) based on admission type and diagnosis
    los = []
    for adm_type, diagnosis in zip(admission_types, primary_diagnoses):
        base_los = 5  # baseline LOS

        # Adjust based on admission type
        if adm_type == 'Emergency':
            base_los += random.randint(0, 5)
        elif adm_type == 'Urgent':
            base_los += random.randint(0, 3)

        # Adjust based on diagnosis
        if diagnosis in ['I50.9', 'J44.9', 'N17.9']:  # more severe conditions
            base_los += random.randint(2, 7)

        # Add some random variation
        final_los = max(1, int(np.random.normal(base_los, 2)))
        los.append(final_los)

    # Generate number of previous admissions (slightly higher for certain conditions)
    prev_admissions = []
    for diagnosis in primary_diagnoses:
        if diagnosis in ['I50.9', 'J44.9', 'E11.9']:  # chronic conditions
            base_prev = random.randint(1, 5)
        else:
            base_prev = random.randint(0, 3)
        prev_admissions.append(base_prev)

    # Generate comorbidity count
    comorbidities = []
    for age, diagnosis in zip(ages, primary_diagnoses):
        base_comorbid = 1
        if age > 65:
            base_comorbid += random.randint(1, 3)
        if diagnosis in ['I50.9', 'E11.9', 'J44.9']:  # chronic conditions
            base_comorbid += random.randint(1, 2)
        comorbidities.append(base_comorbid)

    # Generate emergency visits in past year
    emergency_visits = []
    for prev_adm, diagnosis in zip(prev_admissions, primary_diagnoses):
        base_emergency = prev_adm // 2  # about half of previous admissions were emergency
        if diagnosis in ['I50.9', 'J44.9']:  # conditions prone to emergencies
            base_emergency += random.randint(0, 2)
        emergency_visits.append(base_emergency)

    # Generate insurance types
    insurance_types = np.random.choice(
        ['Medicare', 'Medicaid', 'Private', 'Self-pay'],
        size=n_patients,
        p=[0.4, 0.2, 0.35, 0.05]
    )

    # Generate readmission target (based on risk factors)
    readmitted = []
    for age, los, prev_adm, comorbid, diagnosis in zip(
        ages, los, prev_admissions, comorbidities, primary_diagnoses
    ):
        risk_score = 0

        # Age factor
        if age > 75:
            risk_score += 0.2
        elif age > 65:
            risk_score += 0.1

        # Length of stay factor
        if los > 7:
            risk_score += 0.15

        # Previous admissions factor
        if prev_adm > 2:
            risk_score += 0.2

        # Comorbidity factor
        if comorbid > 2:
            risk_score += 0.15

        # Diagnosis factor
        if diagnosis in ['I50.9', 'J44.9', 'N17.9']:
            risk_score += 0.2

        # Calculate probability and determine readmission
        prob_readmission = min(0.8, risk_score)  # cap at 80% probability
        readmitted.append(1 if random.random() < prob_readmission else 0)

    # Create DataFrame
    data = pd.DataFrame({
        'patient_id': patient_ids,
        'age': ages,
        'gender': genders,
        'insurance_type': insurance_types,
        'admission_type': admission_types,
        'diagnosis_code': primary_diagnoses,
        'diagnosis_description': diagnosis_descriptions,
        'length_of_stay': los,
        'num_previous_admissions': prev_admissions,
        'num_comorbidities': comorbidities,
        'emergency_visits_past_year': emergency_visits,
        'readmitted': readmitted
    })

    # Add some missing values to make it more realistic
    for col in ['num_previous_admissions', 'emergency_visits_past_year', 'num_comorbidities']:
        mask = np.random.random(len(data)) < 0.05  # 5% missing values
        data.loc[mask, col] = np.nan

    return data

def main():
    # Generate the dataset with 10,000 records
    df = generate_hospital_data(n_patients=10000)

    # Save to CSV
    df.to_csv('hospital_readmissions_large.csv', index=False)

    # Display first few rows and basic statistics
    print("\nFirst few rows of the large dataset:")
    print(df.head())

    print("\nDataset Summary:")
    print("-" * 50)
    print(f"Total number of records: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    print(f"Readmission rate: {df['readmitted'].mean():.2%}")

    print("\nFeature Statistics:")
    print("-" * 50)
    print(df.describe())

    print("\nMissing Values:")
    print("-" * 50)
    print(df.isnull().sum())

if __name__ == "__main__":
    main()
