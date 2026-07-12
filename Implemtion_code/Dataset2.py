

import kagglehub
import pandas as pd
import os
import glob

os.makedirs('./data/kaggle', exist_ok=True)

print("="*80)
print("="*80)

datasets = [
    {
        'name': 'downshift/water-quality-monitoring-dataset',
        'filename': 'water_quality_monitoring.csv',
        'description': 'Water Quality Monitoring (time series)',
        'relevance': 'HIGH - Temporal water quality data'
    },
    {
        'name': 'vbmokin/ammonium-prediction-in-river-water',
        'filename': 'river_water_quality.csv',
        'description': 'River Water Quality with Ammonium',
        'relevance': 'HIGH - River-specific data'
    },
    {
        'name': 'adityakadiwal/water-potability',
        'filename': 'water_potability.csv',
        'description': 'Water Quality & Potability',
        'relevance': 'MEDIUM - pH, DO, turbidity'
    },
    {
        'name': 'mssmartypants/water-quality',
        'filename': 'water_quality_full.csv',
        'description': 'Comprehensive Water Quality',
        'relevance': 'MEDIUM - Multiple parameters'
    },
    {
        'name': 'khushikyad001/water-pollution-and-disease',
        'filename': 'water_pollution.csv',
        'description': 'Water Pollution Indicators',
        'relevance': 'MEDIUM - Pollution sources'
    }
]

downloaded = []
failed = []

for dataset in datasets:
    print(f"\n📥 {dataset['description']}")
    print(f"   Dataset: {dataset['name']}")
    print(f"   Relevance: {dataset['relevance']}")

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
                'relevance': dataset['relevance'],
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

print(f"\n✅ Downloaded: {len(downloaded)} datasets")
for d in downloaded:
    print(f"\n📄 {d['filename']}")
    print(f"   Description: {d['description']}")
    print(f"   Relevance: {d['relevance']}")
    print(f"   Records: {d['rows']:,}")
    print(f"   Columns: {d['columns']}")
    print(f"   Variables: {', '.join(d['columns_list'][:5])}...")

print(f"\n❌ Failed: {len(failed)} datasets")
for f in failed:
    print(f"   - {f}")

print("\n📁 Files saved in: ./data/kaggle/")