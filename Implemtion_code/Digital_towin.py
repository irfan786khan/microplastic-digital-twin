"""
DIGITAL TWIN ENVIRONMENT USING PINN MODEL
MICROPLASTIC POLLUTION DIGITAL TWIN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DIGITAL TWIN ENVIRONMENT USING PINN")
print("MICROPLASTIC POLLUTION")
print("="*80)


data_paths = {
    'graph_nodes': './data/graph_nodes.csv',
    'graph_edges': './data/graph_edges_combined.csv',
    'clean_data': './data/clean_spatio_temporal_dataset.csv',
    'collected_microplastic': './data/kaggle/collected_microplastic_concentration.csv',
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



if 'graph_nodes' in datasets:
    unified_df = datasets['graph_nodes'].copy()
else:
    unified_df = pd.DataFrame()

print(f"   Base dataset: {len(unified_df):,} records")

# Add features from clean data
if 'clean_data' in datasets:
    for col in ['temperature', 'turbidity', 'ph', 'dissolved_oxygen', 'salinity', 'discharge']:
        if col in datasets['clean_data'].columns and col not in unified_df.columns:
            if len(datasets['clean_data']) >= len(unified_df):
                unified_df[col] = datasets['clean_data'][col].values[:len(unified_df)]
                print(f"   Added: {col}")



available_features = ['concentration', 'turbidity', 'temperature', 'ph', 'salinity', 'dissolved_oxygen', 'discharge']

feature_cols = []
for col in available_features:
    if col in unified_df.columns:
        feature_cols.append(col)
        print(f"   ✅ Found: {col}")
    else:
        if col == 'turbidity':
            unified_df['turbidity'] = np.random.uniform(1, 20, len(unified_df))
            feature_cols.append('turbidity')
            print(f"   🔄 Created: {col}")
        elif col == 'temperature':
            unified_df['temperature'] = np.random.uniform(10, 25, len(unified_df))
            feature_cols.append('temperature')
            print(f"   🔄 Created: {col}")
        elif col == 'ph':
            unified_df['ph'] = np.random.uniform(6.5, 8.5, len(unified_df))
            feature_cols.append('ph')
            print(f"   🔄 Created: {col}")
        elif col == 'salinity':
            unified_df['salinity'] = np.random.uniform(0.1, 1.0, len(unified_df))
            feature_cols.append('salinity')
            print(f"   🔄 Created: {col}")
        elif col == 'dissolved_oxygen':
            unified_df['dissolved_oxygen'] = np.random.uniform(4, 10, len(unified_df))
            feature_cols.append('dissolved_oxygen')
            print(f"   🔄 Created: {col}")
        elif col == 'discharge':
            unified_df['discharge'] = np.random.uniform(50, 200, len(unified_df))
            feature_cols.append('discharge')
            print(f"   🔄 Created: {col}")

target_col = 'concentration'
if target_col not in unified_df.columns:
    unified_df['concentration'] = (
        2 + 0.3 * unified_df['turbidity'] + 0.1 * unified_df['temperature'] +
        np.random.randn(len(unified_df)) * 0.5
    )
    unified_df['concentration'] = np.maximum(unified_df['concentration'], 0.1)
    print(f"   🔄 Created target: concentration")

print(f"\n   Features ({len(feature_cols)}): {feature_cols}")
print(f"   Target: {target_col}")



sample_size = min(5000, len(unified_df))
np.random.seed(42)
sampled_indices = np.random.choice(len(unified_df), sample_size, replace=False)
sampled_df = unified_df.iloc[sampled_indices].copy().reset_index(drop=True)

print(f"   Sampled: {len(sampled_df):,} records")

X = sampled_df[feature_cols].values
y = sampled_df[target_col].values.reshape(-1, 1)

X = np.nan_to_num(X)
y = np.nan_to_num(y)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

print(f"   X shape: {X_scaled.shape}")
print(f"   y shape: {y_scaled.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print(f"   Training: {len(X_train):,}")
print(f"   Validation: {len(X_val):,}")
print(f"   Testing: {len(X_test):,}")



class PhysicsConstraints:
    def __init__(self):
        self.diffusion = 0.1
        self.sedimentation = 0.05
        self.degradation = 0.01
        self.advection = 0.2

    def compute_loss(self, concentration, features):
        n_features = features.shape[1]

        turbidity = features[:, 1:2] if n_features > 1 else torch.ones_like(concentration)
        temperature = features[:, 2:3] if n_features > 2 else torch.ones_like(concentration)
        salinity = features[:, 3:4] if n_features > 3 else torch.ones_like(concentration)
        discharge = features[:, 4:5] if n_features > 4 else torch.ones_like(concentration)

        turbidity_norm = torch.sigmoid(turbidity)
        salinity_norm = torch.sigmoid(salinity)
        temp_norm = torch.sigmoid(temperature)
        discharge_norm = torch.sigmoid(discharge)

        advection = self.advection * discharge_norm * concentration
        diffusion = self.diffusion * turbidity_norm * concentration
        sedimentation = -self.sedimentation * salinity_norm * concentration
        degradation = -self.degradation * (1 + 0.1 * temp_norm) * concentration

        physics_derivative = advection + diffusion + sedimentation + degradation
        physics_loss = torch.mean(physics_derivative ** 2)

        return physics_loss

physics = PhysicsConstraints()
print("   ✅ Physics constraints ready")



class LightweightPINN(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.physics = PhysicsConstraints()

    def forward(self, x):
        return self.net(x)

    def predict_with_physics(self, x):
        concentration = self.forward(x)
        physics_loss = self.physics.compute_loss(concentration, x)
        return concentration, physics_loss

model = LightweightPINN(input_dim=X_scaled.shape[1], hidden_dim=64)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {total_params:,}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

model = model.to(device)
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)
X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mse = nn.MSELoss()

print("\n   Training progress:")
print("-"*60)

n_epochs = 80
best_val_rmse = float('inf')

for epoch in range(n_epochs):
    model.train()
    concentration, physics_loss = model.predict_with_physics(X_train_t)
    data_loss = mse(concentration, y_train_t)
    total_loss = data_loss + 0.3 * physics_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        model.eval()
        with torch.no_grad():
            val_pred, val_physics = model.predict_with_physics(X_val_t)
            pred_np = val_pred.cpu().numpy()
            true_np = y_val_t.cpu().numpy()

            rmse = np.sqrt(mean_squared_error(true_np, pred_np))
            r2 = r2_score(true_np, pred_np)

            if rmse < best_val_rmse:
                best_val_rmse = rmse
                torch.save(model.state_dict(), './data/best_model_physics_all.pt')
                print(f"   ✅ New best (RMSE: {rmse:.4f}, R²: {r2:.4f})")

            print(f"   Epoch {epoch+1:3d}/{n_epochs} | Data Loss: {data_loss.item():.4f} | Physics: {physics_loss.item():.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")

print("-"*60)
print(f"\n✅ Training complete! Best RMSE: {best_val_rmse:.4f}")



model.load_state_dict(torch.load('./data/best_model_physics_all.pt'))
model.eval()

with torch.no_grad():
    test_pred, test_physics = model.predict_with_physics(X_test_t)
    pred_np = test_pred.cpu().numpy()
    true_np = y_test_t.cpu().numpy()

    test_rmse = np.sqrt(mean_squared_error(true_np, pred_np))
    test_mae = mean_absolute_error(true_np, pred_np)
    test_r2 = r2_score(true_np, pred_np)

print(f"\n   📈 Test Results:")
print(f"   - RMSE: {test_rmse:.4f}")
print(f"   - MAE: {test_mae:.4f}")
print(f"   - R²: {test_r2:.4f}")



class DigitalTwin:
    """
    Digital Twin Environment using trained PINN model
    """

    def __init__(self, model, scaler_X, scaler_y, feature_cols, device='cpu'):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.feature_cols = feature_cols
        self.device = device

        self.model.to(device)
        self.model.eval()

        self.current_state = None
        self.history = []

        # Actions
        self.actions = [
            'reduce_waste',
            'improve_treatment',
            'add_monitoring',
            'increase_filtration',
            'reduce_runoff'
        ]

        self.action_effects = {
            'reduce_waste': {'turbidity': 0.9, 'discharge': 0.95},
            'improve_treatment': {'turbidity': 0.85, 'dissolved_oxygen': 1.05},
            'add_monitoring': {'turbidity': 0.95, 'ph': 1.02},
            'increase_filtration': {'turbidity': 0.8, 'salinity': 0.95},
            'reduce_runoff': {'turbidity': 0.85, 'rainfall': 0.9}
        }

    def reset(self):
        avg_conditions = self.scaler_X.mean_.reshape(1, -1)
        self.current_state = torch.tensor(avg_conditions, dtype=torch.float32).to(self.device)
        self.history = []
        return self._get_observation()

    def _get_observation(self):
        state_np = self.current_state.cpu().numpy()
        return {
            'features': state_np,
            'concentration': self._predict_concentration(state_np)
        }

    def _predict_concentration(self, state):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state_t = torch.tensor(state, dtype=torch.float32).to(self.device)
            else:
                state_t = state
            pred, _ = self.model.predict_with_physics(state_t)
            conc = self.scaler_y.inverse_transform(pred.cpu().numpy())
            return float(conc[0, 0])

    def step(self, action_idx):
        action = self.actions[action_idx]
        effect = self.action_effects.get(action, {})

        state_np = self.current_state.cpu().numpy().copy()

        for feature, multiplier in effect.items():
            if feature in self.feature_cols:
                idx = self.feature_cols.index(feature)
                state_np[0, idx] *= multiplier

        state_np += 0.01 * np.random.randn(*state_np.shape)

        self.current_state = torch.tensor(state_np, dtype=torch.float32).to(self.device)

        conc = self._predict_concentration(state_np)
        reward = -conc / 10.0
        done = conc < 0.2 or conc > 15.0

        self.history.append({
            'action': action,
            'concentration': conc,
            'reward': reward,
            'done': done
        })

        return self._get_observation(), reward, done, {'action': action}

    def get_environment_summary(self):
        return {
            'current_concentration': self._get_observation()['concentration'],
            'num_actions_taken': len(self.history),
            'last_action': self.history[-1]['action'] if self.history else 'none',
            'last_reward': self.history[-1]['reward'] if self.history else 0
        }

    def get_available_actions(self):
        return self.actions



digital_twin = DigitalTwin(
    model=model,
    scaler_X=scaler_X,
    scaler_y=scaler_y,
    feature_cols=feature_cols,
    device=device
)

print("   ✅ Digital Twin created!")


obs = digital_twin.reset()
print(f"\n   Initial concentration: {obs['concentration']:.3f} items/m³")

print("\n   Taking actions...")
for i in range(5):
    action_idx = np.random.randint(len(digital_twin.actions))
    obs, reward, done, info = digital_twin.step(action_idx)
    print(f"   Step {i+1}: Action={info['action']}, Concentration={obs['concentration']:.3f}, Reward={reward:.3f}")

summary = digital_twin.get_environment_summary()
print(f"\n   📊 Summary:")
print(f"   - Current concentration: {summary['current_concentration']:.3f}")
print(f"   - Actions taken: {summary['num_actions_taken']}")
print(f"   - Last action: {summary['last_action']}")



# Save model
torch.save(model.state_dict(), './data/best_model_physics_all.pt')
print("   ✅ Saved: best_model_physics_all.pt")

# Save Digital Twin data
digital_twin_data = {
    'feature_cols': feature_cols,
    'scaler_X': scaler_X,
    'scaler_y': scaler_y,
    'actions': digital_twin.actions,
    'action_effects': digital_twin.action_effects
}

with open('./data/digital_twin.pkl', 'wb') as f:
    pickle.dump(digital_twin_data, f)
print("   ✅ Saved: digital_twin.pkl")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("DIGITAL TWIN - COMPLETE SUMMARY")
print("="*80)

print(f"""
📊 Digital Twin Components:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Component                │ Details                                    │
   ├──────────────────────────┼────────────────────────────────────────────┤
   │ PINN Model               │ {sum(p.numel() for p in model.parameters()):,} params │
   │ Features                 │ {len(feature_cols)}: {', '.join(feature_cols)} │
   │ Actions                  │ {len(digital_twin.actions)}                │
   └──────────────────────────┴────────────────────────────────────────────┘

📈 Model Performance:
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ Metric                   │ Value                                      │
   ├──────────────────────────┼────────────────────────────────────────────┤
   │ Test RMSE                │ {test_rmse:.4f}                            │
   │ Test MAE                 │ {test_mae:.4f}                             │
   │ Test R²                  │ {test_r2:.4f}                              │
   └──────────────────────────┴────────────────────────────────────────────┘

🎮 Available Actions:
   {', '.join(digital_twin.actions)}

📁 Output Files:
   ├── ./data/best_model_physics_all.pt    (PINN Model)
   └── ./data/digital_twin.pkl             (Digital Twin)

🎯 Interface Functions:
   - reset() - Reset environment
   - step(action_idx) - Take action
   - get_environment_summary() - Get current status
   - get_available_actions() - List all actions
""")

print("="*80)
print("✅ DIGITAL TWIN ENVIRONMENT COMPLETE")
print("="*80)



def run_demo(n_steps=10):
    dt = DigitalTwin(model, scaler_X, scaler_y, feature_cols, device)
    dt.reset()

    print("\n   Demo Steps:")
    print("-"*60)

    for step in range(n_steps):
        action_idx = np.random.randint(len(dt.actions))
        obs, reward, done, info = dt.step(action_idx)
        print(f"   Step {step+1:2d}: {info['action']:<20} → Concentration: {obs['concentration']:.3f} | Reward: {reward:.3f}")
        if done:
            break

    print("-"*60)

run_demo(10)

print("\n" + "="*80)
print("✅ DIGITAL TWIN READY FOR REINFORCEMENT LEARNING")
print("="*80)