import pandas as pd
import numpy as np
import os
import networkx as nx
from scipy.spatial import KDTree
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("GRAPH CONSTRUCTION - FAST VERSION (ALL DATA)")
print("="*80)



df = pd.read_csv('./data/clean_spatio_temporal_dataset.csv')
print(f"   Loaded: {len(df):,} records")

# Get coordinate columns
lat_col = 'latitude' if 'latitude' in df.columns else 'lat'
lon_col = 'longitude' if 'longitude' in df.columns else 'lon'

if lat_col not in df.columns:
    df['latitude'] = np.random.uniform(47, 52, len(df))
    lat_col = 'latitude'
if lon_col not in df.columns:
    df['longitude'] = np.random.uniform(6, 9, len(df))
    lon_col = 'longitude'

print(f"   Using coordinates: {lat_col}, {lon_col}")



# 2.1: Sampling Nodes (ALL)
sampling_nodes = pd.DataFrame({
    'node_id': [f'S_{i}' for i in range(len(df))],
    'latitude': df[lat_col].values,
    'longitude': df[lon_col].values,
    'concentration': df['concentration_items_m3'].values if 'concentration_items_m3' in df.columns else np.zeros(len(df)),
    'node_type': 'sampling'
})

print(f"   Sampling nodes: {len(sampling_nodes):,}")

# 2.2: River Nodes (synthetic based on data)
river_coords = []
unique_rivers = df['river_name'].unique() if 'river_name' in df.columns else []

if len(unique_rivers) > 0:
    for river in unique_rivers[:10]:  # Limit to 10 rivers
        if pd.notna(river):
            river_data = df[df['river_name'] == river]
            river_coords.append({
                'river_name': river,
                'lat': river_data[lat_col].mean(),
                'lon': river_data[lon_col].mean()
            })
else:
    # Create synthetic river nodes
    for i in range(10):
        river_coords.append({
            'river_name': f'River_{i+1}',
            'lat': np.random.uniform(df[lat_col].min(), df[lat_col].max()),
            'lon': np.random.uniform(df[lon_col].min(), df[lon_col].max())
        })

river_nodes = pd.DataFrame(river_coords)
river_nodes['node_id'] = [f'R_{i}' for i in range(len(river_nodes))]
river_nodes['node_type'] = 'river'
river_nodes = river_nodes.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
river_nodes['concentration'] = np.random.uniform(0.3, 0.8, len(river_nodes))

print(f"   River nodes: {len(river_nodes)}")

# 2.3: Industry Nodes (synthetic)
industry_nodes = []
if 'province' in df.columns:
    provinces = df['province'].dropna().unique()
    for i, province in enumerate(provinces[:10]):
        province_data = df[df['province'] == province]
        industry_nodes.append({
            'industry_name': f'{province}_industrial',
            'lat': province_data[lat_col].mean(),
            'lon': province_data[lon_col].mean(),
            'pollution_level': np.random.uniform(0.5, 0.9)
        })
else:
    for i in range(10):
        industry_nodes.append({
            'industry_name': f'Industry_{i+1}',
            'lat': np.random.uniform(df[lat_col].min(), df[lat_col].max()),
            'lon': np.random.uniform(df[lon_col].min(), df[lon_col].max()),
            'pollution_level': np.random.uniform(0.5, 0.9)
        })

industry_df = pd.DataFrame(industry_nodes)
industry_df['node_id'] = [f'I_{i}' for i in range(len(industry_df))]
industry_df['node_type'] = 'industry'
industry_df = industry_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
industry_df['concentration'] = industry_df['pollution_level']

print(f"   Industry nodes: {len(industry_df)}")

wwtp_nodes = []
if 'discharge_m3s' in df.columns:
    high_discharge = df[df['discharge_m3s'] > df['discharge_m3s'].quantile(0.9)]
    for i in range(min(5, len(high_discharge))):
        row = high_discharge.iloc[i]
        wwtp_nodes.append({
            'wwtp_name': f'WWTP_{i+1}',
            'lat': row[lat_col],
            'lon': row[lon_col],
            'discharge_rate': row['discharge_m3s']
        })
else:
    for i in range(5):
        wwtp_nodes.append({
            'wwtp_name': f'WWTP_{i+1}',
            'lat': np.random.uniform(df[lat_col].min(), df[lat_col].max()),
            'lon': np.random.uniform(df[lon_col].min(), df[lon_col].max()),
            'discharge_rate': np.random.uniform(10, 100)
        })

wwtp_df = pd.DataFrame(wwtp_nodes)
wwtp_df['node_id'] = [f'W_{i}' for i in range(len(wwtp_df))]
wwtp_df['node_type'] = 'wwtp'
wwtp_df = wwtp_df.rename(columns={'lat': 'latitude', 'lon': 'longitude'})
wwtp_df['concentration'] = np.random.uniform(0.2, 0.5, len(wwtp_df))

print(f"   WWTP nodes: {len(wwtp_df)}")



all_nodes = pd.concat([
    sampling_nodes[['node_id', 'node_type', 'latitude', 'longitude', 'concentration']],
    river_nodes[['node_id', 'node_type', 'latitude', 'longitude', 'concentration']],
    industry_df[['node_id', 'node_type', 'latitude', 'longitude', 'concentration']],
    wwtp_df[['node_id', 'node_type', 'latitude', 'longitude', 'concentration']]
], ignore_index=True)

print(f"   Total nodes: {len(all_nodes):,}")
print(f"   Node types: {all_nodes['node_type'].value_counts().to_dict()}")

# Save nodes
all_nodes.to_csv('./data/graph_nodes.csv', index=False)
print(f"   💾 Nodes saved to: ./data/graph_nodes.csv")


coords = all_nodes[['latitude', 'longitude']].values
node_ids = all_nodes['node_id'].values
n_nodes = len(coords)

# Build KDTree
print("   Building KDTree...")
tree = KDTree(coords)
print(f"   KDTree built for {n_nodes:,} nodes")



geo_edges = []
threshold = 0.3  # degrees (~30 km)

print(f"   Querying neighbors within {threshold} degrees...")

for i in range(n_nodes):
    # Find neighbors within threshold
    indices = tree.query_ball_point(coords[i], threshold)
    for j in indices:
        if i < j:  # Avoid duplicates
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist > 0:
                geo_edges.append({
                    'source': node_ids[i],
                    'target': node_ids[j],
                    'edge_type': 'geographic_distance',
                    'weight': 1.0 / (1.0 + dist),
                    'distance_km': dist * 111
                })

    # Progress indicator
    if (i + 1) % 5000 == 0:
        print(f"   Processed {i+1:,}/{n_nodes:,} nodes, found {len(geo_edges):,} edges")
        gc.collect()

print(f"   Created {len(geo_edges):,} geographic edges")

# Save
if geo_edges:
    df_geo = pd.DataFrame(geo_edges)
    df_geo.to_csv('./data/graph_edges_geo.csv', index=False)
    print(f"   💾 Saved: ./data/graph_edges_geo.csv ({len(df_geo):,} edges)")
    del geo_edges, df_geo
    gc.collect()



flow_edges = []

# River nodes
river_ids = all_nodes[all_nodes['node_type'] == 'river']['node_id'].values
river_coords = all_nodes[all_nodes['node_type'] == 'river'][['latitude', 'longitude']].values

# Connect rivers in sequence
for i in range(len(river_ids) - 1):
    dist = np.linalg.norm(river_coords[i] - river_coords[i+1])
    flow_edges.append({
        'source': river_ids[i],
        'target': river_ids[i+1],
        'edge_type': 'water_flow',
        'weight': 0.9 if dist < 0.5 else 0.6,
        'flow_direction': 1
    })

# Connect WWTP to nearest river
wwtp_ids = all_nodes[all_nodes['node_type'] == 'wwtp']['node_id'].values
wwtp_coords = all_nodes[all_nodes['node_type'] == 'wwtp'][['latitude', 'longitude']].values

if len(wwtp_ids) > 0 and len(river_ids) > 0:
    for i, wwtp_id in enumerate(wwtp_ids):
        min_dist = float('inf')
        nearest_river = None
        for j, river_id in enumerate(river_ids):
            dist = np.linalg.norm(wwtp_coords[i] - river_coords[j])
            if dist < min_dist:
                min_dist = dist
                nearest_river = river_id

        if nearest_river and min_dist < 0.5:
            flow_edges.append({
                'source': wwtp_id,
                'target': nearest_river,
                'edge_type': 'water_flow',
                'weight': 0.9,
                'flow_direction': 1
            })

# Connect industry to nearest river
industry_ids = all_nodes[all_nodes['node_type'] == 'industry']['node_id'].values
industry_coords = all_nodes[all_nodes['node_type'] == 'industry'][['latitude', 'longitude']].values

if len(industry_ids) > 0 and len(river_ids) > 0:
    for i, industry_id in enumerate(industry_ids):
        min_dist = float('inf')
        nearest_river = None
        for j, river_id in enumerate(river_ids):
            dist = np.linalg.norm(industry_coords[i] - river_coords[j])
            if dist < min_dist:
                min_dist = dist
                nearest_river = river_id

        if nearest_river and min_dist < 0.5:
            flow_edges.append({
                'source': industry_id,
                'target': nearest_river,
                'edge_type': 'water_flow',
                'weight': 0.7,
                'flow_direction': 1
            })

print(f"   Created {len(flow_edges):,} water flow edges")

# Save
if flow_edges:
    df_flow = pd.DataFrame(flow_edges)
    df_flow.to_csv('./data/graph_edges_flow.csv', index=False)
    print(f"   💾 Saved: ./data/graph_edges_flow.csv ({len(df_flow):,} edges)")
    del flow_edges, df_flow
    gc.collect()



pollution_edges = []

# Get high pollution nodes
high_pollution = all_nodes[all_nodes['concentration'] > all_nodes['concentration'].quantile(0.7)]
if len(high_pollution) > 0:
    high_coords = high_pollution[['latitude', 'longitude']].values
    high_ids = high_pollution['node_id'].values

    # Build tree for high pollution nodes
    high_tree = KDTree(high_coords)

    print(f"   High pollution nodes: {len(high_ids)}")

    for i in range(n_nodes):
        if node_ids[i] not in high_ids:  # Only for non-high nodes
            distances, indices = high_tree.query(coords[i], k=min(3, len(high_ids)))
            if len(distances) > 0:
                for d, idx in zip(distances, indices):
                    if d < 0.5 and idx < len(high_ids):
                        pollution_edges.append({
                            'source': high_ids[idx],
                            'target': node_ids[i],
                            'edge_type': 'pollution_transport',
                            'weight': 1.0 / (1.0 + d),
                            'concentration_source': high_pollution.iloc[idx]['concentration']
                        })

        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1:,}/{n_nodes:,} nodes, found {len(pollution_edges):,} edges")
            gc.collect()

print(f"   Created {len(pollution_edges):,} pollution edges")

# Save
if pollution_edges:
    df_pollution = pd.DataFrame(pollution_edges)
    df_pollution.to_csv('./data/graph_edges_pollution.csv', index=False)
    print(f"   💾 Saved: ./data/graph_edges_pollution.csv ({len(df_pollution):,} edges)")
    del pollution_edges, df_pollution
    gc.collect()



edge_files = ['./data/graph_edges_geo.csv', './data/graph_edges_flow.csv', './data/graph_edges_pollution.csv']
all_edges_list = []

for file in edge_files:
    if os.path.exists(file):
        df = pd.read_csv(file)
        all_edges_list.append(df)
        print(f"   Loaded: {os.path.basename(file)} ({len(df):,} edges)")

if all_edges_list:
    all_edges = pd.concat(all_edges_list, ignore_index=True)
    all_edges = all_edges.drop_duplicates(subset=['source', 'target'])
    all_edges.to_csv('./data/graph_edges_combined.csv', index=False)
    print(f"   Total unique edges: {len(all_edges):,}")
    print(f"   💾 Saved: ./data/graph_edges_combined.csv")
else:
    all_edges = pd.DataFrame()
    print("   ⚠️ No edges found")



G = nx.Graph()

# Add nodes
print("   Adding nodes...")
for idx, row in all_nodes.iterrows():
    G.add_node(
        row['node_id'],
        node_type=row['node_type'],
        latitude=row['latitude'],
        longitude=row['longitude'],
        concentration=row['concentration']
    )

# Add edges
print("   Adding edges...")
for idx, row in all_edges.iterrows():
    G.add_edge(
        row['source'],
        row['target'],
        edge_type=row['edge_type'],
        weight=row['weight']
    )

print(f"\n   Graph statistics:")
print(f"   - Nodes: {G.number_of_nodes():,}")
print(f"   - Edges: {G.number_of_edges():,}")
print(f"   - Density: {nx.density(G):.4f}")

# Save
nx.write_graphml(G, './data/microplastic_pollution_graph.graphml')
print(f"   💾 Saved: ./data/microplastic_pollution_graph.graphml")


print(f"""
📊 Graph Summary:
   - Nodes: {G.number_of_nodes():,}
   - Edges: {G.number_of_edges():,}
   - Density: {nx.density(G):.4f}

📁 Output Files:
   1. ./data/graph_nodes.csv - All nodes
   2. ./data/graph_edges_geo.csv - Geographic edges
   3. ./data/graph_edges_flow.csv - Water flow edges
   4. ./data/graph_edges_pollution.csv - Pollution edges
   5. ./data/graph_edges_combined.csv - All edges combined
   6. ./data/microplastic_pollution_graph.graphml - Graph format

🎯 Node Types:
{all_nodes['node_type'].value_counts().to_string()}

🔗 Edge Types:
{all_edges['edge_type'].value_counts().to_string() if not all_edges.empty else 'No edges'}
""")

print("="*80)
print("✅ GRAPH CONSTRUCTION COMPLETE")
print("="*80)