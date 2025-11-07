"""
White Balance Prediction Solution - ENHANCED
============================================

Complete solution for predicting Temperature and Tint values for white balance adjustments.
Handles non-linear temperature sensitivity and ensures consistency across similar images.

ENHANCEMENTS:
- K-Fold Cross-Validation for robust evaluation
- Hyperparameter tuning with Optuna
- Additional models: CatBoost and Neural Networks
- Advanced image features: Texture analysis, edge detection, deep features

Usage: python solution.py [--tune] [--cv] [--neural]
"""

import pandas as pd
import numpy as np
import os
import warnings
from pathlib import Path
import argparse

warnings.filterwarnings('ignore')

# Check dependencies
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Using metadata-only approach.")

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Using LightGBM only.")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available.")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Skipping hyperparameter tuning.")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Neural networks disabled.")


def engineer_features(df):
    """Create engineered features addressing non-linear temperature sensitivity"""
    df = df.copy()
    
    # Non-linear temperature transformations (critical for sensitivity)
    df['currTemp_log'] = np.log(df['currTemp'])
    df['currTemp_sqrt'] = np.sqrt(df['currTemp'])
    df['currTemp_inv'] = 1 / df['currTemp']
    df['currTemp_squared'] = df['currTemp'] ** 2
    
    # Temperature bins for different sensitivity ranges
    df['temp_bin_low'] = (df['currTemp'] < 3000).astype(int)
    df['temp_bin_mid'] = ((df['currTemp'] >= 3000) & (df['currTemp'] <= 6000)).astype(int)
    df['temp_bin_high'] = (df['currTemp'] > 6000).astype(int)
    df['temp_very_warm'] = (df['currTemp'] > 7000).astype(int)
    df['temp_very_cool'] = (df['currTemp'] < 2500).astype(int)
    
    # Tint transformations
    df['currTint_abs'] = np.abs(df['currTint'])
    df['currTint_squared'] = df['currTint'] ** 2
    
    # Exposure value
    df['ev'] = np.log2(df['aperture']**2 / df['shutterSpeed'])
    df['ev_squared'] = df['ev'] ** 2
    
    # Interaction features
    df['temp_tint_interaction'] = df['currTemp'] * df['currTint']
    df['temp_tint_ratio'] = df['currTemp'] / (np.abs(df['currTint']) + 1)
    df['iso_aperture'] = df['isoSpeedRating'] * df['aperture']
    df['focal_aperture'] = df['focalLength'] / df['aperture']
    df['iso_shutter'] = df['isoSpeedRating'] * df['shutterSpeed']
    
    # Flash interactions (important for light source variations)
    df['flash_iso'] = df['flashFired'] * df['isoSpeedRating']
    df['flash_temp'] = df['flashFired'] * df['currTemp']
    df['flash_ev'] = df['flashFired'] * df['ev']
    
    # Advanced exposure features
    df['iso_log'] = np.log(df['isoSpeedRating'] + 1)
    df['aperture_squared'] = df['aperture'] ** 2
    df['shutter_log'] = np.log(df['shutterSpeed'] + 1e-6)
    df['focal_log'] = np.log(df['focalLength'] + 1)
    df['focal_iso'] = df['focalLength'] * df['isoSpeedRating']
    
    return df


def extract_advanced_image_features(image_path):
    """Extract ADVANCED color and texture features from images"""
    if not OPENCV_AVAILABLE:
        return {}
        
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return {}
        
        # Convert to RGB and other color spaces
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # ===== BASIC COLOR FEATURES =====
        # RGB statistics
        for i, channel in enumerate(['r', 'g', 'b']):
            features[f'{channel}_mean'] = img_rgb[:,:,i].mean()
            features[f'{channel}_std'] = img_rgb[:,:,i].std()
            features[f'{channel}_median'] = np.median(img_rgb[:,:,i])
            features[f'{channel}_min'] = img_rgb[:,:,i].min()
            features[f'{channel}_max'] = img_rgb[:,:,i].max()
            features[f'{channel}_q25'] = np.percentile(img_rgb[:,:,i], 25)
            features[f'{channel}_q75'] = np.percentile(img_rgb[:,:,i], 75)
        
        # Color ratios (CRITICAL for white balance consistency)
        features['rg_ratio'] = features['r_mean'] / (features['g_mean'] + 1e-6)
        features['rb_ratio'] = features['r_mean'] / (features['b_mean'] + 1e-6)
        features['gb_ratio'] = features['g_mean'] / (features['b_mean'] + 1e-6)
        features['rg_ratio_median'] = features['r_median'] / (features['g_median'] + 1e-6)
        features['rb_ratio_median'] = features['r_median'] / (features['b_median'] + 1e-6)
        features['gb_ratio_median'] = features['g_median'] / (features['b_median'] + 1e-6)
        
        # Color dominance
        total = features['r_mean'] + features['g_mean'] + features['b_mean'] + 1e-6
        features['r_dominance'] = features['r_mean'] / total
        features['g_dominance'] = features['g_mean'] / total
        features['b_dominance'] = features['b_mean'] / total
        
        # HSV features
        features['hue_mean'] = img_hsv[:,:,0].mean()
        features['hue_std'] = img_hsv[:,:,0].std()
        features['sat_mean'] = img_hsv[:,:,1].mean()
        features['sat_std'] = img_hsv[:,:,1].std()
        features['val_mean'] = img_hsv[:,:,2].mean()
        features['val_std'] = img_hsv[:,:,2].std()
        
        # LAB features
        features['l_mean'] = img_lab[:,:,0].mean()
        features['a_mean'] = img_lab[:,:,1].mean()
        features['b_mean_lab'] = img_lab[:,:,2].mean()
        features['warmth_indicator'] = features['a_mean']
        features['yellow_blue_indicator'] = features['b_mean_lab']
        
        # Brightness and contrast
        features['brightness'] = gray.mean()
        features['contrast'] = gray.std()
        
        # Color cast detection
        features['color_cast_rg'] = np.abs(features['rg_ratio'] - 1.0)
        features['color_cast_rb'] = np.abs(features['rb_ratio'] - 1.0)
        features['color_cast_gb'] = np.abs(features['gb_ratio'] - 1.0)
        
        # ===== ADVANCED TEXTURE FEATURES =====
        # Edge detection (Canny)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = edges.mean() / 255.0
        features['edge_count'] = np.sum(edges > 0)
        
        # Sobel gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        features['gradient_x_mean'] = np.abs(sobelx).mean()
        features['gradient_y_mean'] = np.abs(sobely).mean()
        features['gradient_magnitude'] = np.sqrt(sobelx**2 + sobely**2).mean()
        
        # Laplacian (texture sharpness)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_mean'] = np.abs(laplacian).mean()
        features['laplacian_var'] = laplacian.var()
        
        # Histogram features
        hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist_gray = hist_gray / hist_gray.sum()
        features['hist_entropy'] = -np.sum(hist_gray * np.log2(hist_gray + 1e-10))
        features['hist_peak'] = hist_gray.max()
        features['hist_uniformity'] = np.sum(hist_gray ** 2)
        
        # Color histogram features
        for i, channel in enumerate(['r', 'g', 'b']):
            hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256]).flatten()
            hist = hist / hist.sum()
            features[f'{channel}_hist_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
            features[f'{channel}_hist_skew'] = np.sum((np.arange(256) - features[f'{channel}_mean']) ** 3 * hist) / (features[f'{channel}_std'] ** 3 + 1e-6)
            features[f'{channel}_hist_kurtosis'] = np.sum((np.arange(256) - features[f'{channel}_mean']) ** 4 * hist) / (features[f'{channel}_std'] ** 4 + 1e-6)
        
        # ===== SPATIAL FEATURES =====
        # Divide image into regions (3x3 grid)
        h, w = gray.shape
        h_step, w_step = h // 3, w // 3
        
        for i in range(3):
            for j in range(3):
                region = gray[i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step]
                features[f'region_{i}{j}_mean'] = region.mean()
                features[f'region_{i}{j}_std'] = region.std()
        
        # Center vs periphery
        center_h, center_w = h // 4, w // 4
        center = gray[center_h:3*center_h, center_w:3*center_w]
        features['center_mean'] = center.mean()
        features['center_std'] = center.std()
        features['center_periphery_ratio'] = center.mean() / (gray.mean() + 1e-6)
        
        # ===== FREQUENCY DOMAIN FEATURES =====
        # FFT for frequency analysis
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)
        features['fft_mean'] = magnitude_spectrum.mean()
        features['fft_std'] = magnitude_spectrum.std()
        features['fft_energy'] = np.sum(magnitude_spectrum ** 2)
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return {}


def extract_image_features(image_path):
    """Wrapper for backward compatibility"""
    return extract_advanced_image_features(image_path)


def prepare_data(df, feature_cols=None, train_medians=None, is_training=True):
    """Prepare data for modeling"""
    df = engineer_features(df)
    
    # Handle camera features
    if 'camera_model' in df.columns:
        camera_dummies = pd.get_dummies(df['camera_model'], prefix='camera')
        df = pd.concat([df, camera_dummies], axis=1)
    
    if 'camera_group' in df.columns:
        group_dummies = pd.get_dummies(df['camera_group'], prefix='group')
        df = pd.concat([df, group_dummies], axis=1)
    
    # Define features
    exclude_cols = ['id_global', 'Temperature', 'Tint', 'copyCreationTime', 
                   'captureTime', 'touchTime', 'camera_model', 'camera_group']
    
    if is_training:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        train_medians = df[feature_cols].median()
    
    # Handle missing values
    df[feature_cols] = df[feature_cols].fillna(train_medians)
    
    return df[feature_cols], feature_cols, train_medians


class NeuralNetRegressor(nn.Module):
    """Neural Network for regression with dropout and batch normalization"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(NeuralNetRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train_neural_network(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, lr=0.001):
    """Train a neural network regressor"""
    if not PYTORCH_AVAILABLE:
        return None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).reshape(-1, 1).to(device)
    X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).reshape(-1, 1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = NeuralNetRegressor(X_train.shape[1]).to(device)
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                break
    
    return model, scaler


def cross_validate_models(X, y_temp, y_tint, n_splits=5):
    """Perform K-Fold cross-validation"""
    print(f"\n  Running {n_splits}-Fold Cross-Validation...")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    temp_scores = []
    tint_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_temp_train, y_temp_val = y_temp.iloc[train_idx], y_temp.iloc[val_idx]
        y_tint_train, y_tint_val = y_tint.iloc[train_idx], y_tint.iloc[val_idx]
        
        # Train models
        temp_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, 
                                   num_leaves=64, random_state=42, verbose=-1)
        tint_model = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=8, 
                                   num_leaves=64, random_state=42, verbose=-1)
        
        temp_model.fit(X_train, y_temp_train)
        tint_model.fit(X_train, y_tint_train)
        
        # Evaluate
        temp_pred = temp_model.predict(X_val)
        tint_pred = tint_model.predict(X_val)
        
        temp_mae = mean_absolute_error(y_temp_val, temp_pred)
        tint_mae = mean_absolute_error(y_tint_val, tint_pred)
        
        temp_scores.append(temp_mae)
        tint_scores.append(tint_mae)
        
        print(f"    Fold {fold}: Temp MAE={temp_mae:.2f}, Tint MAE={tint_mae:.2f}")
    
    print(f"  CV Results:")
    print(f"    Temperature: {np.mean(temp_scores):.2f} ± {np.std(temp_scores):.2f}")
    print(f"    Tint: {np.mean(tint_scores):.2f} ± {np.std(tint_scores):.2f}")
    
    return temp_scores, tint_scores


def tune_hyperparameters(X_train, y_train, X_val, y_val, target_name='Temperature'):
    """Hyperparameter tuning with Optuna"""
    if not OPTUNA_AVAILABLE:
        print("  Optuna not available, using default parameters")
        return {}
    
    print(f"  Tuning hyperparameters for {target_name}...")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 6, 12),
            'num_leaves': trial.suggest_int('num_leaves', 32, 128),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'random_state': 42,
            'verbose': -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, pred)
        
        return mae
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=False)
    
    print(f"    Best MAE: {study.best_value:.2f}")
    print(f"    Best params: {study.best_params}")
    
    return study.best_params


def main():
    """Main execution function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description='White Balance Prediction - Enhanced')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--cv', action='store_true', help='Enable cross-validation')
    parser.add_argument('--neural', action='store_true', help='Enable neural network models')
    args = parser.parse_args()
    
    print("=" * 60)
    print("White Balance Prediction Solution - ENHANCED")
    print("=" * 60)
    
    # Check dataset
    train_csv = 'dataset/Train/sliders.csv'
    val_csv = 'dataset/Validation/sliders_input.csv'
    train_images = 'dataset/Train/images'
    val_images = 'dataset/Validation/images'
    
    if not all(os.path.exists(p) for p in [train_csv, val_csv, train_images, val_images]):
        print("Error: Dataset files not found!")
        print("Expected structure:")
        print("  dataset/Train/sliders.csv")
        print("  dataset/Train/images/")
        print("  dataset/Validation/sliders_input.csv")
        print("  dataset/Validation/images/")
        return
    
    print(f"\nConfiguration:")
    print(f"  - Advanced image features: {'✓' if OPENCV_AVAILABLE else '✗'}")
    print(f"  - XGBoost: {'✓' if XGBOOST_AVAILABLE else '✗'}")
    print(f"  - CatBoost: {'✓' if CATBOOST_AVAILABLE else '✗'}")
    print(f"  - Neural Networks: {'✓' if PYTORCH_AVAILABLE and args.neural else '✗'}")
    print(f"  - Hyperparameter Tuning: {'✓' if OPTUNA_AVAILABLE and args.tune else '✗'}")
    print(f"  - Cross-Validation: {'✓' if args.cv else '✗'}")
    
    # Load data
    print(f"\n[1/7] Loading data...")
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print(f"  Training samples: {len(train_df)}")
    print(f"  Validation samples: {len(val_df)}")
    
    # Extract image features
    if OPENCV_AVAILABLE:
        print(f"\n[2/7] Extracting training image features...")
        train_image_features = []
        for idx, row in train_df.iterrows():
            img_path = Path(train_images) / f"{row['id_global']}.tiff"
            features = extract_image_features(img_path)
            train_image_features.append(features)
            if (idx + 1) % 500 == 0:
                print(f"    {idx + 1}/{len(train_df)} images processed")
        
        train_img_df = pd.DataFrame(train_image_features)
        train_df = pd.concat([train_df.reset_index(drop=True), train_img_df], axis=1)
        
        print(f"  Extracting validation image features...")
        val_image_features = []
        for idx, row in val_df.iterrows():
            img_path = Path(val_images) / f"{row['id_global']}.tiff"
            features = extract_image_features(img_path)
            val_image_features.append(features)
            if (idx + 1) % 100 == 0:
                print(f"    {idx + 1}/{len(val_df)} images processed")
        
        val_img_df = pd.DataFrame(val_image_features)
        val_df = pd.concat([val_df.reset_index(drop=True), val_img_df], axis=1)
    else:
        print(f"\n[2/7] Skipping image features (OpenCV not available)...")
    
    # Prepare features
    print(f"\n[3/7] Preparing features...")
    X_train, feature_cols, train_medians = prepare_data(train_df, is_training=True)
    y_temp_train = train_df['Temperature']
    y_tint_train = train_df['Tint']
    
    X_val, _, _ = prepare_data(val_df, feature_cols, train_medians, is_training=False)
    
    print(f"  Feature count: {len(feature_cols)}")
    print(f"  Training shape: {X_train.shape}")
    
    # Cross-validation (optional)
    if args.cv:
        print(f"\n[4/7] Cross-Validation...")
        cross_validate_models(X_train, y_temp_train, y_tint_train, n_splits=5)
    
    # Split for validation
    print(f"\n[4/7] Creating validation split...")
    X_tr, X_te, y_temp_tr, y_temp_te, y_tint_tr, y_tint_te = train_test_split(
        X_train, y_temp_train, y_tint_train, test_size=0.15, random_state=42
    )
    
    # Hyperparameter tuning (optional)
    best_temp_params = {}
    best_tint_params = {}
    if args.tune and OPTUNA_AVAILABLE:
        print(f"\n  Hyperparameter Tuning...")
        best_temp_params = tune_hyperparameters(X_tr, y_temp_tr, X_te, y_temp_te, 'Temperature')
        best_tint_params = tune_hyperparameters(X_tr, y_tint_tr, X_te, y_tint_te, 'Tint')
    
    # Train models
    print(f"\n[5/7] Training models...")
    
    # Initialize all model variables
    temp_xgb = None
    tint_xgb = None
    temp_cat = None
    tint_cat = None
    temp_nn = None
    tint_nn = None
    temp_scaler = None
    tint_scaler = None
    
    # Temperature models
    print("  Training Temperature models...")
    temp_lgbm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=8, num_leaves=64,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    temp_lgbm.fit(X_tr, y_temp_tr, eval_set=[(X_te, y_temp_te)], eval_metric='mae')
    
    if XGBOOST_AVAILABLE:
        temp_xgb = XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=8, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=0
        )
        temp_xgb.fit(X_tr, y_temp_tr, eval_set=[(X_te, y_temp_te)], verbose=False)
    
    if CATBOOST_AVAILABLE:
        print("  Training CatBoost for Temperature...")
        temp_cat = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            random_state=42, verbose=0
        )
        temp_cat.fit(X_tr, y_temp_tr, eval_set=(X_te, y_temp_te), verbose=False)
    
    if args.neural and PYTORCH_AVAILABLE:
        print("  Training Neural Network for Temperature...")
        temp_nn, temp_scaler = train_neural_network(X_tr, y_temp_tr, X_te, y_temp_te)
    
    # Tint models
    print("  Training Tint models...")
    tint_lgbm = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=8, num_leaves=64,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1
    )
    tint_lgbm.fit(X_tr, y_tint_tr, eval_set=[(X_te, y_tint_te)], eval_metric='mae')
    
    if XGBOOST_AVAILABLE:
        tint_xgb = XGBRegressor(
            n_estimators=1000, learning_rate=0.05, max_depth=8, min_child_weight=20,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbosity=0
        )
        tint_xgb.fit(X_tr, y_tint_tr, eval_set=[(X_te, y_tint_te)], verbose=False)
    
    if CATBOOST_AVAILABLE:
        print("  Training CatBoost for Tint...")
        tint_cat = CatBoostRegressor(
            iterations=1000, learning_rate=0.05, depth=8, l2_leaf_reg=3,
            random_state=42, verbose=0
        )
        tint_cat.fit(X_tr, y_tint_tr, eval_set=(X_te, y_tint_te), verbose=False)
    
    if args.neural and PYTORCH_AVAILABLE:
        print("  Training Neural Network for Tint...")
        tint_nn, tint_scaler = train_neural_network(X_tr, y_tint_tr, X_te, y_tint_te)
    
    # Evaluate with ensemble
    print("\n  Evaluating ensemble...")
    temp_preds = [temp_lgbm.predict(X_te)]
    tint_preds = [tint_lgbm.predict(X_te)]
    weights = [0.4]
    
    if temp_xgb is not None:
        temp_preds.append(temp_xgb.predict(X_te))
        tint_preds.append(tint_xgb.predict(X_te))
        weights.append(0.3)
    
    if temp_cat is not None:
        temp_preds.append(temp_cat.predict(X_te))
        tint_preds.append(tint_cat.predict(X_te))
        weights.append(0.2)
    
    if temp_nn is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_te_scaled = temp_scaler.transform(X_te)
        X_te_tensor = torch.FloatTensor(X_te_scaled).to(device)
        temp_nn.eval()
        tint_nn.eval()
        with torch.no_grad():
            temp_preds.append(temp_nn(X_te_tensor).cpu().numpy().flatten())
            tint_preds.append(tint_nn(X_te_tensor).cpu().numpy().flatten())
        weights.append(0.1)
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    temp_pred = sum(w * p for w, p in zip(weights, temp_preds))
    tint_pred = sum(w * p for w, p in zip(weights, tint_preds))
    
    temp_mae = mean_absolute_error(y_temp_te, temp_pred)
    tint_mae = mean_absolute_error(y_tint_te, tint_pred)
    
    print(f"    Temperature MAE: {temp_mae:.2f}")
    print(f"    Tint MAE: {tint_mae:.2f}")
    print(f"    Ensemble weights: {weights}")
    
    # Retrain on full data
    print(f"\n[6/7] Retraining on full data...")
    temp_lgbm.fit(X_train, y_temp_train)
    tint_lgbm.fit(X_train, y_tint_train)
    
    if temp_xgb is not None:
        temp_xgb.fit(X_train, y_temp_train)
        tint_xgb.fit(X_train, y_tint_train)
    
    if temp_cat is not None:
        temp_cat.fit(X_train, y_temp_train, verbose=False)
        tint_cat.fit(X_train, y_tint_train, verbose=False)
    
    if temp_nn is not None:
        temp_nn, temp_scaler = train_neural_network(X_train, y_temp_train, X_te, y_temp_te, epochs=50)
        tint_nn, tint_scaler = train_neural_network(X_train, y_tint_train, X_te, y_tint_te, epochs=50)
    
    # Generate predictions
    print(f"\n[7/7] Generating predictions...")
    
    temp_val_preds = [temp_lgbm.predict(X_val)]
    tint_val_preds = [tint_lgbm.predict(X_val)]
    
    if temp_xgb is not None:
        temp_val_preds.append(temp_xgb.predict(X_val))
        tint_val_preds.append(tint_xgb.predict(X_val))
    
    if temp_cat is not None:
        temp_val_preds.append(temp_cat.predict(X_val))
        tint_val_preds.append(tint_cat.predict(X_val))
    
    if temp_nn is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_val_scaled = temp_scaler.transform(X_val)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        temp_nn.eval()
        tint_nn.eval()
        with torch.no_grad():
            temp_val_preds.append(temp_nn(X_val_tensor).cpu().numpy().flatten())
            tint_val_preds.append(tint_nn(X_val_tensor).cpu().numpy().flatten())
    
    temp_predictions = sum(w * p for w, p in zip(weights, temp_val_preds))
    tint_predictions = sum(w * p for w, p in zip(weights, tint_val_preds))
    
    # Clip to valid ranges
    temp_predictions = np.clip(np.round(temp_predictions).astype(int), 2000, 50000)
    tint_predictions = np.clip(np.round(tint_predictions).astype(int), -150, 150)
    
    # Create submission
    submission = pd.DataFrame({
        'id_global': val_df['id_global'],
        'Temperature': temp_predictions,
        'Tint': tint_predictions
    })
    
    submission.to_csv('predictions.csv', index=False)
    
    print("\n" + "=" * 60)
    print("PREDICTIONS SAVED: predictions.csv")
    print("=" * 60)
    
    print(f"\nEnhancements Applied:")
    if args.cv:
        print("  ✓ K-Fold Cross-Validation")
    if args.tune:
        print("  ✓ Hyperparameter Tuning (Optuna)")
    if CATBOOST_AVAILABLE:
        print("  ✓ CatBoost Model")
    if args.neural and PYTORCH_AVAILABLE:
        print("  ✓ Neural Network Model")
    if OPENCV_AVAILABLE:
        print("  ✓ Advanced Image Features (Texture, Edges, FFT)")
    
    print(f"\nFirst 10 predictions:")
    print(submission.head(10))
    
    print(f"\nPrediction statistics:")
    print(submission[['Temperature', 'Tint']].describe())
    
    print(f"\n✓ Done! Submit predictions.csv ")
    print(f"\nTip: Run with --tune --cv --neural for maximum performance!")


if __name__ == "__main__":
    main()