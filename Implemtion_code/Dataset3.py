

import kagglehub
import pandas as pd
import os
import glob

os.makedirs('./data/kaggle', exist_ok=True)



datasets_to_download = [
    {
        'name': 'brsdincer/marine-microplastic-on-world-density-noaa',
        'filename': 'marine_microplastic_noaa.csv',
        'description': 'MARINE MICROPLASTIC DENSITY - NOAA (CORE DATA)',
        'priority': 'CRITICAL'
    },
    {
        'name': 'piaoya/plastic-recycling-codes',
        'filename': 'plastic_recycling_codes.csv',
        'description': 'POLYMER TYPES - Recycling Codes',
        'priority': 'CRITICAL'
    },
    {
        'name': 'prajwaldongre/global-plastic-waste-2023-a-country-wise-analysis',
        'filename': 'global_plastic_waste_2023.csv',
        'description': 'Global Plastic Waste by Country',
        'priority': 'HIGH'
    },
    {
        'name': 'imtkaggleteam/plastic-pollution',
        'filename': 'plastic_pollution.csv',
        'description': 'Plastic Pollution Data',
        'priority': 'MEDIUM'
    },
    {
        'name': 'kkhandekar/mismanaged-plastic-waste-around-the-world',
        'filename': 'mismanaged_plastic_waste.csv',
        'description': 'Mismanaged Plastic Waste',
        'priority': 'MEDIUM'
    }
]

downloaded = []
failed = []

for dataset in datasets_to_download:
    print(f"\n📥 {dataset['priority']} PRIORITY: {dataset['description']}")
    print(f"   Dataset: {dataset['name']}")

    try:
        path = kagglehub.dataset_download(dataset['name'])
        csv_files = glob.glob(f"{path}/*.csv")

        if csv_files:
            df = pd.read_csv(csv_files[0])
            save_path = f"./data/kaggle/{dataset['filename']}"
            df.to_csv(save_path, index=False)

            downloaded.append({
                'filename': dataset['filename'],
                'description': dataset['description'],
                'priority': dataset['priority'],
                'rows': len(df),
                'columns': len(df.columns),
                'columns_list': list(df.columns)
            })

            print(f"   ✅ Saved: {len(df):,} records, {len(df.columns)} columns")
            print(f"      Columns: {', '.join(list(df.columns)[:5])}...")
        else:
            failed.append(dataset['name'])
            print("   ❌ No CSV found")

    except Exception as e:
        failed.append(dataset['name'])
        print(f"   ❌ Error: {str(e)[:100]}")

# Summary
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)

