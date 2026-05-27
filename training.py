import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import StandardScaler
from model import STMLP

# --- Configuration ---
INPUT_LEN = 12       # Past 12 steps (1 hour)
OUTPUT_LEN = 12      # Future 12 steps (1 hour)
INPUT_DIM = 2        # CHANGED: Now using 2 features (Flow + Hour of Day)
STATIC_DIM = 2       # Latitude, Longitude
EMBED_DIM = 64
EPOCHS = 300         # INCREASED: To prevent underfitting
BATCH_SIZE = 32      # Increased batch size for stability
REG_LAMBDA = 0       # CHANGED: Set to 0 to prevent over-smoothing

def load_data():
    print("Loading and processing datasets...")
    
    # 1. Load Traffic Series (Dynamic Features)
    df_ts = pd.read_csv('traffic_time_series.csv')
    
    # Feature Engineering: Add Normalized Hour (0-1)
    df_ts['timestamp'] = pd.to_datetime(df_ts['timestamp'])
    df_ts['hour_norm'] = df_ts['timestamp'].dt.hour / 23.0
    
    # Pivot Flow: (Time x Sensors)
    pivot_flow = df_ts.pivot(index='timestamp', columns='sensor_id', values='flow').ffill().bfill()
    # Pivot Hour: (Time x Sensors) - repeats for all sensors
    pivot_hour = df_ts.pivot(index='timestamp', columns='sensor_id', values='hour_norm').ffill().bfill()
    
    # Ensure consistent sensor order
    sensor_ids = sorted(pivot_flow.columns)
    pivot_flow = pivot_flow[sensor_ids]
    pivot_hour = pivot_hour[sensor_ids]
    
    # Normalize Flow
    scaler = StandardScaler()
    flow_values = scaler.fit_transform(pivot_flow.values)
    hour_values = pivot_hour.values # Already 0-1
    
    # Combine into (Time, Nodes, Features=2)
    # Stack along last axis
    data_combined = np.stack([flow_values, hour_values], axis=-1)
    
    # 2. Load Sensors Metadata (Static Features)
    df_sensors = pd.read_csv('sensors.csv')
    df_sensors = df_sensors.set_index('sensor_id').reindex(sensor_ids).reset_index()
    static_feats = df_sensors[['latitude', 'longitude']].values
    static_scaler = StandardScaler()
    static_feats_norm = static_scaler.fit_transform(static_feats)
    
    # 3. Load Adjacency Edges (Graph Structure)
    # We load this to compute Laplacian, even if REG_LAMBDA is 0 (for future use)
    df_edges = pd.read_csv('adjacency_edges.csv')
    adj_matrix = np.zeros((len(sensor_ids), len(sensor_ids)))
    sensor_to_idx = {sid: i for i, sid in enumerate(sensor_ids)}
    
    for _, row in df_edges.iterrows():
        if row['source_sensor'] in sensor_to_idx and row['target_sensor'] in sensor_to_idx:
            i, j = sensor_to_idx[row['source_sensor']], sensor_to_idx[row['target_sensor']]
            adj_matrix[i, j] = row['connection_weight']
            adj_matrix[j, i] = row['connection_weight']
            
    # Compute Laplacian
    degree = np.sum(adj_matrix, axis=1)
    # Avoid division by zero
    degree[degree == 0] = 1e-5 
    laplacian = np.diag(degree) - adj_matrix
    d_inv_sqrt = np.diag(np.power(degree, -0.5))
    norm_laplacian = np.dot(np.dot(d_inv_sqrt, laplacian), d_inv_sqrt)
    
    # 4. Load Sequences (Train Splits)
    df_seq = pd.read_csv('traffic_sequences.csv')
    train_indices = df_seq[df_seq['dataset_split'] == 'train']['history_start_step'].unique()
    
    # Create Batches
    def create_dataset(indices):
        X, Y = [], []
        for idx in indices:
            if idx + INPUT_LEN + OUTPUT_LEN < len(data_combined):
                # Input: Past window (Flow + Hour)
                X.append(data_combined[idx : idx + INPUT_LEN])
                # Target: Future window (Flow Only) - We only predict flow
                # So we take index 0 of the feature dimension
                Y.append(data_combined[idx + INPUT_LEN : idx + INPUT_LEN + OUTPUT_LEN, :, 0])
        return np.array(X), np.array(Y)

    train_X, train_Y = create_dataset(train_indices)
    
    print(f"Training Samples: {train_X.shape[0]}")
    print(f"Input Shape: {train_X.shape} (Time, Nodes, Features)")
    
    return {
        'train_x': torch.tensor(train_X, dtype=torch.float32),
        'train_y': torch.tensor(train_Y, dtype=torch.float32), # Target is just Flow
        'static_feat': torch.tensor(static_feats_norm, dtype=torch.float32),
        'laplacian': torch.tensor(norm_laplacian, dtype=torch.float32),
        'num_nodes': len(sensor_ids),
        'scaler': scaler,
        'static_scaler': static_scaler,
        'sensor_ids': sensor_ids
    }

def train_model():
    data = load_data()
    
    # Save Scalers for App
    os.makedirs('saved_models', exist_ok=True)
    with open('saved_models/scaler.pkl', 'wb') as f:
        pickle.dump(data['scaler'], f)
    with open('saved_models/static_scaler.pkl', 'wb') as f:
        pickle.dump(data['static_scaler'], f)
    with open('saved_models/sensor_ids.pkl', 'wb') as f:
        pickle.dump(data['sensor_ids'], f)

    # Initialize Model
    # Note: input_dim=2 because we added 'hour', but output_len predicts just 'flow'
    model = STMLP(
        num_nodes=data['num_nodes'],
        input_len=INPUT_LEN,
        input_dim=INPUT_DIM, 
        static_dim=STATIC_DIM,
        embed_dim=EMBED_DIM,
        output_len=OUTPUT_LEN
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Prepare static tensors
    static_feat_base = data['static_feat']
    laplacian = data['laplacian']
    train_x = data['train_x']
    train_y = data['train_y']
    
    print(f"Starting Training for {EPOCHS} epochs...")
    model.train()
    
    num_samples = len(train_x)
    
    for epoch in range(EPOCHS):
        permutation = torch.randperm(num_samples)
        epoch_loss = 0
        
        for i in range(0, num_samples, BATCH_SIZE):
            indices = permutation[i : i + BATCH_SIZE]
            batch_x = train_x[indices]
            batch_y = train_y[indices]
            
            # Expand static features to match batch size
            curr_batch = batch_x.size(0)
            batch_static = static_feat_base.unsqueeze(0).expand(curr_batch, -1, -1)
            
            optimizer.zero_grad()
            
            # Forward Pass
            output = model(batch_x, batch_static) # Output is (Batch, Output_Len, Nodes)
            
            # Loss Calculation
            mse_loss = criterion(output, batch_y)
            
            # Optional: Graph Regularization (Currently Disabled via REG_LAMBDA=0)
            if REG_LAMBDA > 0:
                pred_mean = output.mean(dim=[0, 1])
                reg_loss = torch.matmul(pred_mean.unsqueeze(0), torch.matmul(laplacian, pred_mean.unsqueeze(1)))
                loss = mse_loss + (REG_LAMBDA * reg_loss.squeeze())
            else:
                loss = mse_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            avg_loss = epoch_loss / (num_samples / BATCH_SIZE)
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), 'saved_models/st_mlp.pth')
    print("Training Complete. Model saved.")

if __name__ == "__main__":
    train_model()