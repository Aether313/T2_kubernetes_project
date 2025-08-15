import pandas as pd
import numpy as np
import os

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # 3. Clean IMDb Votes (remove commas and convert to numeric)
    if 'IMDb Votes' in df.columns:
        df['IMDb Votes'] = df['IMDb Votes'].replace(',', '', regex=True)
        df['IMDb Votes'] = pd.to_numeric(df['IMDb Votes'], errors='coerce')

    # 4. Clean Boxoffice: remove $, commas, convert to float; fill missing with 'Unknown'
    def clean_boxoffice(val):
        if pd.isnull(val) or val == '':
            return 'Unknown'
        val = str(val).replace('$', '').replace(',', '')
        try:
            return float(val)
        except:
            return 'Unknown'

    if 'Boxoffice' in df.columns:
        df['Boxoffice'] = df['Boxoffice'].apply(clean_boxoffice)

    # 5. Standardize Runtime values into minutes
    def convert_runtime(val):
        if pd.isnull(val):
            return np.nan
        val = str(val).lower()
        if '< 30' in val:
            return 25
        elif '1-2 hour' in val:
            return 90
        elif '30-60' in val:
            return 45
        elif '> 2' in val:
            return 130
        elif 'varies' in val:
            return np.nan
        try:
            return int(val)
        except:
            return np.nan

    if 'Runtime' in df.columns:
        df['Runtime'] = df['Runtime'].apply(convert_runtime)

    # 6. Fill all object-type columns with 'Unknown'
    for col in df.select_dtypes(include='object'):
        df[col] = df[col].fillna('Unknown')

    # 7. Fill numeric columns with median
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    # 8. Keep only selected columns
    columns_to_keep = [
        'Title', 'Genre', 'IMDb Score', 'IMDb Votes', 'Runtime',
        'Languages', 'Director', 'Writer', 'Tags', 'Hidden Gem Score',
        'Boxoffice', 'Production House', 'Country Availability'
    ]
    columns_present = [col for col in columns_to_keep if col in df.columns]
    return df[columns_present]

def clean_file(input_path, output_path):
    # Read the CSV file with UTF-8-SIG encoding
    df = pd.read_csv(input_path, encoding='utf-8-sig')
    df_cleaned = clean_df(df)

    # 9. Save the cleaned CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_cleaned.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Cleaning complete. File saved as {output_path}")

if __name__ == "__main__":
    print("Running cleaning script...")
    input_file = os.environ.get("INPUT_FILE", "/data/imdb_rating.csv")
    output_file = os.environ.get("OUTPUT_FILE", "/data/imdb_rating_cleaned.csv")
    clean_file(input_file, output_file)

