

import kagglehub
import pandas as pd
import os
import glob

os.makedirs('./data/kaggle', exist_ok=True)



polymer_datasets = [
    {
        'name': 'victorsabanzagil/polymers',
        'filename': 'polymers_data.csv',
        'description': 'Polymers dataset - general polymer properties',
        'priority': 'HIGH'
    },
    {
        'name': 'linyeping/extra-dataset-with-smilestgpidpolimers-class',
        'filename': 'polymer_smiles_data.csv',
        'description': 'Polymers with SMILES, Tg, PID, Class',
        'priority': 'HIGH'
    },
    {
        'name': 'ahsanneural/microplastics-food-to-human-bloodstream',
        'filename': 'microplastics_human.csv',
        'description': 'Microplastics: Food to Human Bloodstream',
        'priority': 'MEDIUM'
    },
    {
        'name': 'oleggromov/polymer-tg-density-excerpt',
        'filename': 'polymer_tg_density.csv',
        'description': 'Polymer Tg and Density data',
        'priority': 'MEDIUM'
    },
    {
        'name': 'tasmim/external-polymer-data',
        'filename': 'external_polymer_data.csv',
        'description': 'External Polymer Data',
        'priority': 'LOW'
    }
]

downloaded = []
failed = []

for dataset in polymer_datasets:
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

