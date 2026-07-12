import pandas as pd
import numpy as np
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA CLEANING USING ALL 24 DATASETS")
print("="*80)




all_datasets = {}
dataset_paths = {
    # Hugging Face Datasets
    'wind_10m': './data/huggingface/Antajitters_WindSpeed_10m.csv',
    'wind_50m': './data/huggingface/Antajitters_WindSpeed_50m.csv',
    'wind_100m': './data/huggingface/Antajitters_WindSpeed_100m.csv',
    'groundwater': './data/huggingface/2imi9_OlmoEarth-v1-FT-Karst-Groundwater-Base.csv',

    # Kaggle Datasets
    'microplastics_human': './data/kaggle/microplastics_human.csv',
    'china_water_pollution': './data/kaggle/china_water_pollution.csv',
    'water_pollution': './data/kaggle/water_pollution.csv',
    'polymer_tg_density': './data/kaggle/polymer_tg_density.csv',
    'water_quality_monitoring': './data/kaggle/water_quality_monitoring_complete.csv',
    'river_water_quality': './data/kaggle/river_water_quality.csv',
    'land_use_statistics': './data/kaggle/land_use_statistics.csv',
    'polymers_data': './data/kaggle/polymers_data.csv',
    'river_discharge': './data/kaggle/river_discharge_time_series.csv',
    'mismanaged_waste': './data/kaggle/mismanaged_plastic_waste.csv',
    'microplastic': './data/kaggle/synthetic_microplastic_concentration.csv',
    'synthetic_discharge': './data/kaggle/synthetic_global_river_discharge.csv',
    'water_potability': './data/kaggle/water_potability.csv',
    'polymer_smiles': './data/kaggle/polymer_smiles_data.csv',
    'plastic_waste_2023': './data/kaggle/global_plastic_waste_2023.csv',
    'plastic_pollution': './data/kaggle/plastic_pollution.csv',
    'water_quality_monitoring_orig': './data/kaggle/water_quality_monitoring.csv',
    'external_polymer': './data/kaggle/external_polymer_data.csv',
    'water_quality_full': './data/kaggle/water_quality_full.csv'
}

loaded = 0
failed = []

for name, path in dataset_paths.items():
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_datasets[name] = df
            loaded += 1
            print(f"   ✅ {name}: {len(df):,} records, {len(df.columns)} columns")
        else:
            failed.append(name)
            print(f"   ❌ {name}: File not found")
    except Exception as e:
        failed.append(name)
        print(f"   ❌ {name}: {str(e)[:50]}")

print(f"\n📊 Loaded: {loaded} datasets, Failed: {len(failed)}")



if 'microplastic' in all_datasets:
    df_unified = all_datasets['microplastic'].copy()
    print(f"   Base dataset: microplastic ({len(df_unified):,} records)")
else:
    df_unified = pd.DataFrame()
    print("   ❌ No base dataset found")

# Extract water quality variables
def extract_variables(df, source_name, columns_to_extract):
    """Extract specific columns from a dataset"""
    extracted = {}
    for col in columns_to_extract:
        if col in df.columns:
            # Rename to avoid conflicts
            new_name = f"{col}_{source_name}" if col in df_unified.columns else col
            extracted[new_name] = df[col].values
    return extracted

# Water Quality Monitoring
if 'water_quality_monitoring' in all_datasets:
    df = all_datasets['water_quality_monitoring']
    cols = ['Temperature', 'Dissolved_Oxygen', 'pH', 'Salinity', 'Turbidity',
            'Average_Water_Speed', 'Chlorophyll']
    for col in cols:
        if col in df.columns:
            new_name = f"wq_{col.lower()}" if col in df_unified.columns else col.lower()
            # Align by date if possible
            if 'timestamp' in df.columns:
                df_unified['date'] = pd.to_datetime(df_unified['date'])
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Merge on date
                df_merged = pd.merge(
                    df_unified,
                    df[['timestamp', col]],
                    left_on='date',
                    right_on='timestamp',
                    how='left'
                )
                df_unified[col.lower()] = df_merged[col]
            elif len(df) >= len(df_unified):
                df_unified[col.lower()] = df[col].values[:len(df_unified)]

# China Water Pollution (has spatial data!)
if 'china_water_pollution' in all_datasets:
    df = all_datasets['china_water_pollution']
    # Extract spatial and water quality variables
    spatial_cols = ['Latitude', 'Longitude', 'Province', 'City']
    wq_cols = ['Water_Temperature_C', 'pH', 'Dissolved_Oxygen_mg_L', 'Turbidity_NTU']

    for col in spatial_cols + wq_cols:
        if col in df.columns:
            new_name = f"china_{col.lower()}" if col in df_unified.columns else col.lower()
            if len(df) >= len(df_unified):
                df_unified[new_name] = df[col].values[:len(df_unified)]

# Discharge data
if 'river_discharge' in all_datasets:
    df = all_datasets['river_discharge']
    discharge_cols = ['discharge_armley', 'discharge_kildwick', 'river_level_snaygill']
    for col in discharge_cols:
        if col in df.columns:
            if len(df) >= len(df_unified):
                df_unified[f"discharge_{col}"] = df[col].values[:len(df_unified)]

# Synthetic discharge
if 'synthetic_discharge' in all_datasets:
    df = all_datasets['synthetic_discharge']
    if 'discharge_m3s' in df.columns:
        if len(df) >= len(df_unified):
            df_unified['discharge_m3s'] = df['discharge_m3s'].values[:len(df_unified)]

# Wind data
for wind_name in ['wind_10m', 'wind_50m', 'wind_100m']:
    if wind_name in all_datasets:
        df = all_datasets[wind_name]
        if 'speedavg' in df.columns:
            if len(df) >= len(df_unified):
                df_unified[f'{wind_name}_speed'] = df['speedavg'].values[:len(df_unified)]
        if 'temperatureavg' in df.columns:
            if len(df) >= len(df_unified):
                df_unified[f'{wind_name}_temp'] = df['temperatureavg'].values[:len(df_unified)]

# Groundwater data
if 'groundwater' in all_datasets:
    df = all_datasets['groundwater']
    if 'lat' in df.columns and 'lon' in df.columns:
        if len(df) >= len(df_unified):
            df_unified['groundwater_lat'] = df['lat'].values[:len(df_unified)]
            df_unified['groundwater_lon'] = df['lon'].values[:len(df_unified)]

# Polymer types from multiple sources
polymer_sources = ['microplastics_human', 'polymer_tg_density', 'polymer_smiles', 'external_polymer']
for source in polymer_sources:
    if source in all_datasets:
        df = all_datasets[source]
        if 'abbreviation' in df.columns or 'abr' in df.columns:
            poly_col = 'abbreviation' if 'abbreviation' in df.columns else 'abr'
            if len(df) >= len(df_unified):
                df_unified[f'{source}_polymer'] = df[poly_col].values[:len(df_unified)]
        if 'density_g_per_cm3' in df.columns or 'density' in df.columns:
            dens_col = 'density_g_per_cm3' if 'density_g_per_cm3' in df.columns else 'density'
            if len(df) >= len(df_unified):
                df_unified[f'{source}_density'] = df[dens_col].values[:len(df_unified)]

# Land use
if 'land_use_statistics' in all_datasets:
    df = all_datasets['land_use_statistics']
    land_cols = ['cultivated_land_km2', 'arable_land_km2', 'forest_km2', 'urban_km2']
    for col in land_cols:
        if col in df.columns:
            if len(df) >= len(df_unified):
                df_unified[f'land_{col}'] = df[col].values[:len(df_unified)]

# Waste data
waste_sources = ['mismanaged_waste', 'plastic_waste_2023']
for source in waste_sources:
    if source in all_datasets:
        df = all_datasets[source]
        if 'total_mismanagedplasticwaste_2019_milliont' in df.columns:
            col = 'total_mismanagedplasticwaste_2019_milliont'
            if len(df) >= len(df_unified):
                df_unified[f'{source}_waste'] = df[col].values[:len(df_unified)]

print(f"\n   ✅ Unified dataset: {len(df_unified):,} records, {len(df_unified.columns)} columns")



print("\n📅 STEP 3: Creating temporal features...")

if 'date' in df_unified.columns:
    df_unified['date'] = pd.to_datetime(df_unified['date'], errors='coerce')

    # Extract temporal features
    df_unified['year'] = df_unified['date'].dt.year
    df_unified['month'] = df_unified['date'].dt.month
    df_unified['day'] = df_unified['date'].dt.day
    df_unified['day_of_year'] = df_unified['date'].dt.dayofyear
    df_unified['quarter'] = df_unified['date'].dt.quarter
    df_unified['weekday'] = df_unified['date'].dt.weekday

    # Cyclical encoding
    df_unified['month_sin'] = np.sin(2 * np.pi * df_unified['month'] / 12)
    df_unified['month_cos'] = np.cos(2 * np.pi * df_unified['month'] / 12)
    df_unified['day_sin'] = np.sin(2 * np.pi * df_unified['day_of_year'] / 365)
    df_unified['day_cos'] = np.cos(2 * np.pi * df_unified['day_of_year'] / 365)

    print("   ✅ Temporal features created")

# ============================================================================
# STEP 4: FEATURE ENGINEERING
# ============================================================================

print("\n⚙️ STEP 4: Feature Engineering...")

# 1. Water Quality Index (WQI)
if all(col in df_unified.columns for col in ['ph', 'dissolved_oxygen', 'turbidity']):
    ph_score = np.clip((df_unified['ph'] - 6.5) / (8.5 - 6.5), 0, 1)
    do_score = np.clip(df_unified['dissolved_oxygen'] / 8, 0, 1)
    turb_score = 1 - np.clip(df_unified['turbidity'] / 30, 0, 1)
    df_unified['wqi'] = (ph_score * 0.3 + do_score * 0.4 + turb_score * 0.3)

# 2. Pollution Index
if all(col in df_unified.columns for col in ['turbidity', 'dissolved_oxygen']):
    df_unified['pollution_index'] = (
        (df_unified['turbidity'] / df_unified['turbidity'].max()) * 0.5 +
        (1 - df_unified['dissolved_oxygen'] / df_unified['dissolved_oxygen'].max()) * 0.5
    )

# 3. Lag features for concentration
if 'concentration_items_m3' in df_unified.columns:
    df_unified['concentration_lag1'] = df_unified['concentration_items_m3'].shift(1)
    df_unified['concentration_lag7'] = df_unified['concentration_items_m3'].shift(7)
    df_unified['concentration_lag30'] = df_unified['concentration_items_m3'].shift(30)
    df_unified['concentration_rolling_mean7'] = df_unified['concentration_items_m3'].rolling(7).mean()
    df_unified['concentration_rolling_std7'] = df_unified['concentration_items_m3'].rolling(7).std()
    df_unified['concentration_pct_change'] = df_unified['concentration_items_m3'].pct_change() * 100

# 4. Interaction features
if all(col in df_unified.columns for col in ['turbidity', 'concentration_items_m3']):
    df_unified['turbidity_concentration_ratio'] = df_unified['turbidity'] / (df_unified['concentration_items_m3'] + 0.001)

if all(col in df_unified.columns for col in ['temperature', 'concentration_items_m3']):
    df_unified['temp_concentration_interaction'] = df_unified['temperature'] * df_unified['concentration_items_m3']

# 5. Polymer dummy encoding
polymer_cols = [col for col in df_unified.columns if 'polymer' in col.lower()]
if polymer_cols:
    for col in polymer_cols:
        if df_unified[col].dtype == 'object':
            dummies = pd.get_dummies(df_unified[col], prefix=f'polymer_{col}', dummy_na=True)
            df_unified = pd.concat([df_unified, dummies], axis=1)

# 6. Spatial features
if all(col in df_unified.columns for col in ['latitude', 'longitude']):
    df_unified['lat_lon_interaction'] = df_unified['latitude'] * df_unified['longitude']
    df_unified['spatial_cluster'] = np.floor(df_unified['latitude'] * 10) * 10 + np.floor(df_unified['longitude'] * 10)

print(f"   ✅ Feature engineering complete: {len(df_unified.columns)} columns")




# Select numeric columns only
numeric_cols = df_unified.select_dtypes(include=[np.number]).columns
df_numeric = df_unified[numeric_cols].copy()

missing_before = df_numeric.isna().sum().sum()
print(f"   Missing values before imputation: {missing_before:,}")

if missing_before > 0:
    # KNN Imputation
    imputer = KNNImputer(n_neighbors=5, weights='distance')
    imputed_array = imputer.fit_transform(df_numeric)
    df_imputed = pd.DataFrame(imputed_array, columns=numeric_cols, index=df_unified.index)

    # Replace numeric columns
    for col in numeric_cols:
        df_unified[col] = df_imputed[col]

    missing_after = df_unified.isna().sum().sum()
    print(f"   Missing values after imputation: {missing_after:,}")
    print(f"   ✅ KNN imputation complete")



# Define columns to normalize (exclude target and identifiers)
exclude_cols = ['date', 'year', 'month', 'day', 'day_of_year', 'quarter', 'weekday',
                'month_sin', 'month_cos', 'day_sin', 'day_cos', 'polymer_type']
target_col = 'concentration_items_m3'

numeric_cols = df_unified.select_dtypes(include=[np.number]).columns
numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
numeric_cols = [col for col in numeric_cols if col != target_col]

# Split features and target
X = df_unified[numeric_cols].copy()
y = df_unified[target_col] if target_col in df_unified.columns else None

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=numeric_cols, index=df_unified.index)

# Replace scaled columns
for col in numeric_cols:
    df_unified[col] = X_scaled_df[col]

print(f"   ✅ Normalized {len(numeric_cols)} features using StandardScaler")



if target_col in df_unified.columns and len(df_unified) > 100:
    # Prepare data
    X = df_unified[numeric_cols].copy()
    y = df_unified[target_col].copy()

    # Remove columns with all NaN or zero variance
    X = X.dropna(axis=1, how='all')
    X = X.loc[:, X.var() > 0.001]

    # Remove high correlation (optional)
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr_cols = [col for col in upper.columns if any(upper[col] > 0.95)]
    X = X.drop(columns=high_corr_cols, errors='ignore')

    # Feature selection
    k = min(30, len(X.columns))
    if k > 1:
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        selector.fit(X, y)

        selected_mask = selector.get_support()
        selected_features = X.columns[selected_mask].tolist()
        selected_features = selected_features + [target_col]

        print(f"   Selected {len(selected_features)} features out of {len(X.columns)}")
        print(f"   Top 10 features: {selected_features[:10]}")

        # Keep only selected columns
        df_unified = df_unified[selected_features].copy()



# Remove rows with NaN
df_unified = df_unified.dropna()
print(f"   Final records: {len(df_unified):,}")

# Sort by date if available
if 'date' in df_unified.columns:
    df_unified = df_unified.sort_values('date')
    print("   ✅ Sorted by date")



df_unified.to_csv('./data/clean_spatio_temporal_dataset.csv', index=False)

print(f"\n✅ Clean dataset saved!")
print(f"   Records: {len(df_unified):,}")
print(f"   Features: {len(df_unified.columns)}")
print(f"   Target: {target_col}")

print("\n📊 Final columns:")
for i, col in enumerate(df_unified.columns[:15], 1):
    print(f"   {i}. {col}")
if len(df_unified.columns) > 15:
    print(f"   ... and {len(df_unified.columns)-15} more")


if target_col in df_unified.columns:
    print(f"\n📊 Target Variable: {target_col}")
    print(df_unified[target_col].describe())

print("\n📊 Missing values after cleanup:")
print(df_unified.isna().sum()[df_unified.isna().sum() > 0])

print("\n" + "="*80)
print("✅ DATA CLEANING COMPLETE")
print("="*80)
print("\n📁 Output: ./data/clean_spatio_temporal_dataset.csv")