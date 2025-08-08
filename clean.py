import pandas as pd
import numpy as np

def main():
    # Read the CSV file with UTF-8-SIG encoding
    df = pd.read_csv('imdb_rating.csv', encoding='utf-8-sig')

    # 1. Remove duplicate rows
    df.drop_duplicates(inplace=True)

    # 2. Drop columns that are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # 3. Clean IMDb Votes (remove commas and convert to numeric)
    df['IMDb Votes'] = df['IMDb Votes'].replace(',', '', regex=True)
    df['IMDb Votes'] = pd.to_numeric(df['IMDb Votes'], errors='coerce')

    # 4. Clean Boxoffice: remove $, commas, convert to float; fill missing with 'Unknown'
    def clean_boxoffice(val):
        if pd.isnull(val) or val == '':
            return 'Unknown'
        val = val.replace('$', '').replace(',', '')
        try:
            return float(val)
        except:
            return 'Unknown'

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
    df_cleaned = df[columns_to_keep]

    # 9. Save the cleaned CSV
    df_cleaned.to_csv('imdb_rating_cleaned.csv', index=False, encoding='utf-8-sig')
    print("Cleaning complete. File saved as imdb_rating_cleaned.csv")

if __name__ == "__main__":
    print("Running cleaning script...")
    main()

