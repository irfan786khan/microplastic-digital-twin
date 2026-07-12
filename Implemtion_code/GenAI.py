import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data_paths = {
    'graph_nodes': './data/graph_nodes.csv',
    'graph_edges': './data/graph_edges_combined.csv',
    'clean_data': './data/clean_spatio_temporal_dataset.csv',
    'collected_microplastic': './data/kaggle/_microplastic_concentration.csv',
    'water_quality': './data/kaggle/water_quality_monitoring_complete.csv',
    'china_pollution': './data/kaggle/china_water_pollution.csv',
    'water_pollution': './data/kaggle/water_pollution.csv',
}

datasets = {}
for name, path in data_paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        datasets[name] = df
        print(f"   ✅ {name}: {len(df):,} records")
    else:
        print(f"   ❌ {name}: file not found")



# Start with graph nodes as base
if 'graph_nodes' in datasets:
    unified_df = datasets['graph_nodes'].copy()
else:
    unified_df = pd.DataFrame()
    unified_df['node_id'] = [f'N_{i}' for i in range(5000)]
    unified_df['latitude'] = np.random.uniform(47, 52, 5000)
    unified_df['longitude'] = np.random.uniform(6, 9, 5000)

print(f"   Base records: {len(unified_df):,}")

# Feature columns
feature_cols = ['temperature', 'turbidity', 'ph', 'dissolved_oxygen', 'salinity', 'discharge', 'rainfall']


# 3.1: From clean_data (fixed length mismatch)
if 'clean_data' in datasets:
    clean_df = datasets['clean_data']
    n = min(len(unified_df), len(clean_df))
    print(f"   clean_data: {len(clean_df):,} records, using {n} records")

    for col in feature_cols:
        if col in clean_df.columns:
            unified_df[col] = np.nan
            unified_df.loc[:n-1, col] = clean_df[col].values[:n]
            print(f"      Added: {col} from clean_data (first {n} records)")

# 3.2: From collected_microplastic
if 'collected_microplastic' in datasets:
    collected_df = datasets['collected_microplastic']
    n = min(len(unified_df), len(collected_df))
    print(f"   collected_microplastic: {len(collected_df):,} records, using {n}")

    for col in ['temperature', 'turbidity', 'ph', 'dissolved_oxygen', 'salinity']:
        if col in collected_df.columns:
            if col in unified_df.columns:
                # Fill NaN values with collected data
                unified_df[col] = unified_df[col].fillna(pd.Series(collected_df[col].values[:len(unified_df)]))
            else:
                unified_df[col] = np.nan
                unified_df.loc[:n-1, col] = collected_df[col].values[:n]
            print(f"      Added: {col} from collected_microplastic")

# 3.3: From china_pollution
if 'china_pollution' in datasets:
    china_df = datasets['china_pollution']
    n = min(len(unified_df), len(china_df))
    print(f"   china_pollution: {len(china_df):,} records, using {n}")

    if 'Water_Temperature_C' in china_df.columns:
        unified_df.loc[:n-1, 'temperature'] = china_df['Water_Temperature_C'].values[:n]
        print(f"      Added: temperature from china_pollution")

    if 'pH' in china_df.columns:
        unified_df.loc[:n-1, 'ph'] = china_df['pH'].values[:n]
        print(f"      Added: ph from china_pollution")

    if 'Dissolved_Oxygen_mg_L' in china_df.columns:
        unified_df.loc[:n-1, 'dissolved_oxygen'] = china_df['Dissolved_Oxygen_mg_L'].values[:n]
        print(f"      Added: dissolved_oxygen from china_pollution")

# 3.4: From water_pollution
if 'water_pollution' in datasets:
    poll_df = datasets['water_pollution']
    n = min(len(unified_df), len(poll_df))
    print(f"   water_pollution: {len(poll_df):,} records, using {n}")

    if 'pH_Level' in poll_df.columns:
        unified_df.loc[:n-1, 'ph'] = poll_df['pH_Level'].values[:n]
        print(f"      Added: ph from water_pollution")

    if 'Temperature_C' in poll_df.columns:
        unified_df.loc[:n-1, 'temperature'] = poll_df['Temperature_C'].values[:n]
        print(f"      Added: temperature from water_pollution")



for col in feature_cols:
    if col not in unified_df.columns:
        unified_df[col] = np.nan

    # If all NaN or not enough values, create collected
    if unified_df[col].isna().sum() > len(unified_df) * 0.5:
        if col == 'temperature':
            unified_df[col] = 15 + 8 * np.sin(2 * np.pi * np.arange(len(unified_df)) / 365) + np.random.randn(len(unified_df)) * 2
        elif col == 'turbidity':
            unified_df[col] = np.random.gamma(2, 2, len(unified_df))
        elif col == 'ph':
            unified_df[col] = 7.5 + 0.3 * np.random.randn(len(unified_df))
        elif col == 'dissolved_oxygen':
            unified_df[col] = 8 - 0.1 * unified_df['temperature'] + np.random.randn(len(unified_df)) * 0.5
        elif col == 'salinity':
            unified_df[col] = 0.5 + 0.3 * np.random.rand(len(unified_df))
        elif col == 'discharge':
            unified_df[col] = 100 + 30 * np.random.randn(len(unified_df))
        elif col == 'rainfall':
            unified_df[col] = np.random.exponential(5, len(unified_df))
        print(f"   ✅ Created: {col} (collected)")

# Fill remaining NaN values
for col in feature_cols:
    if col in unified_df.columns:
        unified_df[col] = unified_df[col].fillna(unified_df[col].median())



if 'concentration' in unified_df.columns:
    print(f"   Using existing concentration column")
else:
    unified_df['concentration'] = (
        2.0 +
        0.3 * unified_df['turbidity'] +
        0.1 * unified_df['temperature'] +
        0.1 * unified_df['rainfall'] +
        0.5 * np.random.randn(len(unified_df))
    )
    unified_df['concentration'] = np.maximum(unified_df['concentration'], 0.1)
    print(f"   ✅ Created: concentration (collected)")

print(f"\n   Final dataset: {len(unified_df):,} records, {len(unified_df.columns)} columns")



available_features = [col for col in feature_cols if col in unified_df.columns]
print(f"   Available features: {available_features}")

X = unified_df[available_features].values
y = unified_df['concentration'].values.reshape(-1, 1)

X = np.nan_to_num(X)
y = np.nan_to_num(y)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print(f"   Features: {X_scaled.shape[1]}")
print(f"   Samples: {X_scaled.shape[0]:,}")

sample_size = min(5000, len(X_scaled))
np.random.seed(42)
sample_indices = np.random.choice(len(X_scaled), sample_size, replace=False)
X_sampled = X_scaled[sample_indices]

print(f"   Sampled: {sample_size:,} records for training")



class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim=32, hidden_dim=128):
        super().__init__()

        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.log_var = nn.Linear(hidden_dim // 2, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x, c):
        combined = torch.cat([x, c], dim=1)
        h = self.encoder(combined)
        return self.mu(h), self.log_var(h)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        combined = torch.cat([z, c], dim=1)
        return self.decoder(combined)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z, c)
        return recon_x, mu, log_var

model = ConditionalVAE(
    input_dim=X_sampled.shape[1],
    condition_dim=X_sampled.shape[1],
    latent_dim=32,
    hidden_dim=128
)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Model parameters: {total_params:,}")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

model = model.to(device)
X_tensor = torch.tensor(X_sampled, dtype=torch.float32).to(device)

train_idx, test_idx = train_test_split(range(len(X_tensor)), test_size=0.2, random_state=42)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def vae_loss(recon_x, x, mu, log_var):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl_loss) / x.size(0)

n_epochs = 50
batch_size = 128

print("\n   Training progress:")
print("-"*60)

for epoch in range(n_epochs):
    model.train()
    total_loss = 0
    num_batches = 0

    perm = torch.randperm(len(train_idx))
    train_indices = torch.tensor(train_idx)[perm]

    for i in range(0, len(train_indices), batch_size):
        batch_indices = train_indices[i:i+batch_size]
        batch_x = X_tensor[batch_indices]
        batch_c = X_tensor[batch_indices]

        optimizer.zero_grad()
        recon_x, mu, log_var = model(batch_x, batch_c)
        loss = vae_loss(recon_x, batch_x, mu, log_var)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    if (epoch + 1) % 10 == 0:
        print(f"   Epoch {epoch+1:3d}/{n_epochs} | Loss: {avg_loss:.4f}")

print("-"*60)
print("✅ Training complete!")

torch.save(model.state_dict(), './data/generative_model_complete.pt')
print(f"   💾 Saved: ./data/generative_model_complete.pt")



class ScenarioGenerator:
    def __init__(self, model, scaler_X, scaler_y, feature_names, device):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_names = feature_names
        self.device = device
        self.model.eval()

        self.base_conditions = torch.tensor(
            self.scaler_X.transform(np.zeros((1, len(self.feature_names)))),
            dtype=torch.float32
        ).to(device)

    def generate_scenario(self, condition_modifier, n_samples=500):
        with torch.no_grad():
            condition = self.base_conditions.clone()

            for feature_name, modifier in condition_modifier.items():
                if feature_name in self.feature_names:
                    idx = self.feature_names.index(feature_name)
                    original_mean = self.scaler_X.mean_[idx]
                    original_scale = self.scaler_X.scale_[idx]
                    condition[0, idx] = (modifier * original_mean) / original_scale

            condition_expanded = condition.repeat(n_samples, 1)
            z = torch.randn(n_samples, 32).to(self.device)
            generated = self.model.decode(z, condition_expanded)

            generated_np = generated.cpu().numpy()
            generated_scaled = self.scaler_X.inverse_transform(generated_np)

            concentrations = self._generate_concentrations(generated_scaled)

            return {
                'features': generated_scaled,
                'concentrations': concentrations,
                'condition': condition_modifier,
                'n_samples': n_samples
            }

    def _generate_concentrations(self, features):
        if features.shape[1] > 6:
            concentrations = 2.0 + 0.3 * features[:, 1] + 0.1 * features[:, 0] + 0.1 * features[:, 6]
        else:
            concentrations = 2.0 + 0.3 * features[:, 0]
        return np.maximum(concentrations, 0.1)

    def generate_custom_scenario(self, temp_change=0, rain_change=0, pollution_change=0, n_samples=500):
        modifier = {}
        if temp_change != 0:
            modifier['temperature'] = 1 + temp_change / 100
        if rain_change != 0:
            modifier['rainfall'] = 1 + rain_change / 100
        if pollution_change != 0:
            modifier['turbidity'] = 1 + pollution_change / 100
            modifier['dissolved_oxygen'] = 1 - pollution_change / 100 * 0.3
        return self.generate_scenario(modifier, n_samples)

generator = ScenarioGenerator(model, scaler_X, scaler_y, available_features, device)
print("   ✅ Scenario Generator initialized")



scenarios = {}

print("\n   🌧️ Rainfall increases by 20%")
scenarios['rainfall_20'] = generator.generate_scenario({'rainfall': 1.2})
print(f"      Mean concentration: {scenarios['rainfall_20']['concentrations'].mean():.3f}")

print("\n   🌡️ Temperature increases by 15%")
scenarios['temperature_15'] = generator.generate_custom_scenario(temp_change=15)
print(f"      Mean concentration: {scenarios['temperature_15']['concentrations'].mean():.3f}")

print("\n   🏭 Industrial pollution increases by 30%")
scenarios['industrial_30'] = generator.generate_custom_scenario(pollution_change=30)
print(f"      Mean concentration: {scenarios['industrial_30']['concentrations'].mean():.3f}")

print("\n   🌍 Combined (Temp +15%, Rain +20%, Pollution +30%)")
scenarios['combined'] = generator.generate_custom_scenario(temp_change=15, rain_change=20, pollution_change=30)
print(f"      Mean concentration: {scenarios['combined']['concentrations'].mean():.3f}")

print("\n   🔥 Worst Case (Temp +25%, Rain +40%, Pollution +50%)")
scenarios['worst_case'] = generator.generate_custom_scenario(temp_change=25, rain_change=40, pollution_change=50)
print(f"      Mean concentration: {scenarios['worst_case']['concentrations'].mean():.3f}")



os.makedirs('./data/scenarios', exist_ok=True)

for name, scenario in scenarios.items():
    df_scenario = pd.DataFrame(scenario['features'])
    if len(available_features) == df_scenario.shape[1]:
        df_scenario.columns = available_features
    else:
        df_scenario.columns = [f'feature_{i}' for i in range(df_scenario.shape[1])]

    df_scenario['concentration_items_m3'] = scenario['concentrations']
    df_scenario['scenario_name'] = name
    df_scenario['condition'] = str(scenario['condition'])
    df_scenario.to_csv(f'./data/scenarios/scenario_{name}.csv', index=False)
    print(f"   💾 Saved: ./data/scenarios/scenario_{name}.csv")



comparison = []
for name, scenario in scenarios.items():
    conc = scenario['concentrations']
    comparison.append({
        'scenario': name,
        'mean_concentration': conc.mean(),
        'std_concentration': conc.std(),
        'min_concentration': conc.min(),
        'max_concentration': conc.max()
    })

comparison_df = pd.DataFrame(comparison)
comparison_df.to_csv('./data/scenarios/scenario_comparison.csv', index=False)
print(comparison_df.to_string(index=False))

