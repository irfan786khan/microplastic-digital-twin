

import kagglehub
import pandas as pd
import os
import glob

os.makedirs('./data/kaggle', exist_ok=True)



# Most relevant datasets from search results
datasets_to_download = [
    {
        'name': 'downshift/water-quality-monitoring-dataset',
        'filename': 'water_quality_monitoring_complete.csv',
        'description': 'Water Quality Monitoring Dataset',
        'relevance': 'HIGH - Water quality time series'
    },
    {
        'name': 'thomaswrightanderson/river-aire-discharge-time-series',
        'filename': 'river_discharge_time_series.csv',
        'description': 'River Aire Discharge Time Series',
        'relevance': 'HIGH - River discharge data'
    },
    {
        'name': 'karltarbet/global-monthly-river-discharge-data-set-rivdis',
        'filename': 'global_river_discharge.csv',
        'description': 'Global Monthly River Discharge Data Set',
        'relevance': 'HIGH - Global discharge data'
    },
    {
        'name': 'rajkumarpandey02/land-use-statistics-by-country',
        'filename': 'land_use_statistics.csv',
        'description': 'Land Use Statistics by Country',
        'relevance': 'MEDIUM - Land use data'
    },
    {
        'name': 'khushikyad001/china-water-pollution-monitoring-dataset',
        'filename': 'china_water_pollution.csv',
        'description': 'China Water Pollution Monitoring',
        'relevance': 'HIGH - Water pollution data'
    }
]

downloaded = []
failed = []

for dataset in datasets_to_download:
    print(f"\n📥 {dataset['relevance']}: {dataset['description']}")
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

