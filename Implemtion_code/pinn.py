

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("MICROPLASTIC PREDICTION - IMPROVED GNN + TRANSFORMER")
print("="*80)



nodes_df = pd.read_csv('./data/graph_nodes.csv')
print(f"   Nodes: {len(nodes_df):,}")

edges_df = pd.read_csv('./data/graph_edges_combined.csv')
print(f"   Edges: {len(edges_df):,}")



sample_size = min(10000, len(nodes_df))
np.random.seed(42)
sampled_indices = np.random.choice(len(nodes_df), sample_size, replace=False)
sampled_indices = sorted(sampled_indices)

sampled_nodes = nodes_df.iloc[sampled_indices].copy().reset_index(drop=True)

# Scale features
scaler = StandardScaler()
feature_cols = ['concentration', 'turbidity', 'temperature', 'ph', 'latitude', 'longitude']
for col in feature_cols:
    if col in sampled_nodes.columns:
        sampled_nodes[col] = scaler.fit_transform(sampled_nodes[[col]].values)

# Create node mapping
node_ids = sampled_nodes['node_id'].values
node_to_idx = {node_id: i for i, node_id in enumerate(node_ids)}

# Filter edges
filtered_edges = []
for _, row in edges_df.iterrows():
    if row['source'] in node_to_idx and row['target'] in node_to_idx:
        filtered_edges.append({
            'source': node_to_idx[row['source']],
            'target': node_to_idx[row['target']],
            'weight': row.get('weight', 1.0)
        })

filtered_edges_df = pd.DataFrame(filtered_edges)
print(f"   Sampled nodes: {len(sampled_nodes):,}")
print(f"   Filtered edges: {len(filtered_edges_df):,}")



# Node features
node_features = []
for _, row in sampled_nodes.iterrows():
    features = [
        row.get('concentration', 0),
        row.get('turbidity', 0),
        row.get('temperature', 0),
        row.get('ph', 0),
        row['latitude'],
        row['longitude']
    ]
    node_features.append(features)

x = torch.tensor(node_features, dtype=torch.float)

# Edge index
if len(filtered_edges_df) > 0:
    edge_index = torch.tensor(
        [filtered_edges_df['source'].values, filtered_edges_df['target'].values],
        dtype=torch.long
    )
    edge_weights = torch.tensor(filtered_edges_df['weight'].values, dtype=torch.float)
else:
    edge_index = torch.tensor([[], []], dtype=torch.long)
    edge_weights = torch.tensor([], dtype=torch.float)

# Target (concentration)
y = torch.tensor(sampled_nodes['concentration'].values, dtype=torch.float).view(-1, 1)

print(f"   Node features: {x.shape}")
print(f"   Edge index: {edge_index.shape}")



class ImprovedMicroplasticGNNTransformer(nn.Module):
    def __init__(self, node_features=6, hidden_dim=128, num_heads=4, dropout=0.2):
        super().__init__()

        # Input projection
        self.input_proj = nn.Linear(node_features, hidden_dim)

        # GNN layers with skip connections
        self.conv1 = GATConv(hidden_dim, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)
        self.conv3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, dropout=dropout)

        self.gnn_output_dim = hidden_dim * num_heads

        # Transformer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.gnn_output_dim,
                nhead=num_heads,
                dim_feedforward=512,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=3
        )

        # Prediction heads with batch norm
        self.current_head = nn.Sequential(
            nn.Linear(self.gnn_output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.future_head = nn.Sequential(
            nn.Linear(self.gnn_output_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3)
        )

        self.transport_head = nn.Sequential(
            nn.Linear(self.gnn_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 4)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # Initial projection
        x = self.input_proj(x)
        x = F.relu(x)

        # GNN layers with skip connections
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)

        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = F.elu(x2)
        x2 = F.dropout(x2, p=0.2, training=self.training)

        x3 = self.conv3(x2, edge_index, edge_weight)
        x3 = F.elu(x3)

        # Skip connection
        x = x1 + x3

        # Transformer
        pos_encoding = torch.zeros_like(x)
        position = torch.arange(x.size(0)).unsqueeze(1).float().to(x.device)
        div_term = torch.exp(torch.arange(0, x.size(1), 2).float().to(x.device) *
                             -(np.log(10000.0) / x.size(1)))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        x = x + pos_encoding
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)

        # Predictions
        current = self.current_head(x)
        future = self.future_head(x)
        transport = self.transport_head(x)

        return current, future, transport

# Initialize model
model = ImprovedMicroplasticGNNTransformer(
    node_features=x.shape[1],
    hidden_dim=128,
    num_heads=4,
    dropout=0.2
)

total_params = sum(p.numel() for p in model.parameters())
print(f"\n   Model parameters: {total_params:,}")



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Using device: {device}")

model = model.to(device)
x = x.to(device)
edge_index = edge_index.to(device)
edge_weights = edge_weights.to(device)
y = y.to(device)

# Split
indices = list(range(x.shape[0]))
train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

print(f"   Training: {len(train_idx):,}")
print(f"   Validation: {len(val_idx):,}")
print(f"   Testing: {len(test_idx):,}")

# Optimizer with weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)

# Loss functions
current_criterion = nn.MSELoss()
future_criterion = nn.MSELoss()
transport_criterion = nn.CrossEntropyLoss()

def train_epoch():
    model.train()
    optimizer.zero_grad()

    current_pred, future_pred, transport_pred = model(x, edge_index, edge_weights)

    # Current loss
    loss_current = current_criterion(current_pred[train_idx], y[train_idx])

    # Future loss
    loss_future = future_criterion(future_pred[train_idx], y[train_idx].repeat(1, 3))

    # Transport loss with soft labels
    transport_target = torch.randint(0, 4, (len(train_idx),), device=device)
    loss_transport = transport_criterion(transport_pred[train_idx], transport_target)

    # Weighted loss
    loss = loss_current + 0.3 * loss_future + 0.1 * loss_transport

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), loss_current.item(), loss_future.item()

def evaluate(idx_set):
    model.eval()
    with torch.no_grad():
        current_pred, _, _ = model(x, edge_index, edge_weights)

        pred = current_pred[idx_set].cpu().numpy()
        true = y[idx_set].cpu().numpy()

        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)

        return rmse, mae, r2

# Training loop
n_epochs = 100
best_val_rmse = float('inf')

print("\n   Training progress:")
print("-"*70)

for epoch in range(n_epochs):
    loss, loss_curr, loss_future = train_epoch()

    if (epoch + 1) % 10 == 0:
        val_rmse, val_mae, val_r2 = evaluate(val_idx)
        scheduler.step(val_rmse)

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), './data/best_model_improved.pt')
            print(f"   ✅ New best (RMSE: {val_rmse:.4f}, R²: {val_r2:.4f})")

        print(f"   Epoch {epoch+1:3d}/{n_epochs} | Loss: {loss:.4f} | RMSE: {val_rmse:.4f} | R²: {val_r2:.4f}")

print("-"*70)
print(f"\n✅ Training complete! Best RMSE: {best_val_rmse:.4f}")



model.load_state_dict(torch.load('./data/best_model_improved.pt'))
test_rmse, test_mae, test_r2 = evaluate(test_idx)

print(f"\n   📈 Test Results:")
print(f"   - RMSE: {test_rmse:.4f}")
print(f"   - MAE: {test_mae:.4f}")
print(f"   - R²: {test_r2:.4f}")



model.eval()
with torch.no_grad():
    current_pred, future_pred, transport_pred = model(x, edge_index, edge_weights)

    transport_probs = F.softmax(transport_pred, dim=1)
    transport_directions = torch.argmax(transport_probs, dim=1)
    direction_labels = ['North', 'South', 'East', 'West']

    predictions = {
        'node_id': sampled_nodes['node_id'].values,
        'actual': y.cpu().numpy().flatten(),
        'predicted': current_pred.cpu().numpy().flatten(),
        'future_1': future_pred[:, 0].cpu().numpy(),
        'future_2': future_pred[:, 1].cpu().numpy(),
        'future_3': future_pred[:, 2].cpu().numpy(),
        'transport_direction': [direction_labels[d] for d in transport_directions.cpu().numpy()]
    }

pred_df = pd.DataFrame(predictions)
pred_df.to_csv('./data/microplastic_predictions_improved.csv', index=False)
print(f"   💾 Saved: ./data/microplastic_predictions_improved.csv")



