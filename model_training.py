#!/usr/bin/env python3
# model_training.py - Train and evaluate stuttering classification models with class balancing

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           f1_score, balanced_accuracy_score, precision_recall_fscore_support)
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import seaborn as sns
from tqdm import tqdm

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/model_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train stuttering classification models with class balancing')
    
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory with extracted embeddings')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--model_type', type=str, default='wavlm',
                        choices=['whisper', 'wavlm', 'bestrq', 'combined', 'whisper_large_fixed'],
                        help='Type of embeddings to use')
    parser.add_argument('--split', type=str, default='predefined',
                        choices=['train_test', 'predefined', 'all'],
                        help='How to split the data (predefined=use dataset splits, train_test=manual split)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size for train_test split')
    parser.add_argument('--use_smote', action='store_true', default=True,
                        help='Use SMOTE oversampling')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced learning')
    parser.add_argument('--smote_k_neighbors', type=int, default=3,
                        help='Number of neighbors for SMOTE')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits')
    return parser.parse_args()

def load_data(embeddings_dir, model_type, split='all'):
    """Load embeddings and metadata"""
    # Check if embeddings_dir already includes model_type
    if os.path.basename(embeddings_dir) == model_type or model_type in embeddings_dir:
        model_dir = embeddings_dir
    else:
        model_dir = os.path.join(embeddings_dir, model_type)
    
    if not os.path.exists(model_dir):
        logger.error(f"Embeddings directory for {model_type} not found: {model_dir}")
        return None, {}
    
    # Special handling for predefined split
    if split == 'predefined':
        logger.info("Using predefined splits: train/test/devel from subdirectories")
        
        metadata_frames = []
        for sub_split in ['train', 'test', 'devel']:
            meta_path = os.path.join(model_dir, sub_split, 'embedding_metadata.csv')
            if not os.path.exists(meta_path):
                logger.error(f"Metadata file not found for {sub_split}: {meta_path}")
                exit()
            
            df = pd.read_csv(meta_path)
            df['split'] = sub_split
            metadata_frames.append(df)
            
        metadata_df = pd.concat(metadata_frames, ignore_index=True)
        
        # Load embeddings from all splits
        all_embeddings = {}
        
        for sub_split in ['train', 'test', 'devel']:
            split_dir = os.path.join(model_dir, sub_split)
            
            # Find embedding files
            embedding_files = [f for f in os.listdir(split_dir) if f.endswith('_embeddings.npy')]
            layer_names = [os.path.splitext(f)[0].replace('_embeddings', '') for f in embedding_files]
            
            logger.info(f"Found embeddings for {len(layer_names)} layers in {sub_split} split: {layer_names}")
            
            # Load embeddings for each layer
            for layer, filename in zip(layer_names, embedding_files):
                try:
                    embedding_array = np.load(os.path.join(split_dir, filename))
                    
                    if layer not in all_embeddings:
                        all_embeddings[layer] = []
                    
                    all_embeddings[layer].append(embedding_array)
                    logger.info(f"Loaded {layer} embeddings for {sub_split} split with shape {embedding_array.shape}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Combine embeddings
        for layer in all_embeddings:
            if len(all_embeddings[layer]) > 1:
                all_embeddings[layer] = np.vstack(all_embeddings[layer])
            else:
                all_embeddings[layer] = all_embeddings[layer][0]
            logger.info(f"Combined {layer} embeddings with final shape {all_embeddings[layer].shape}")
        
        return metadata_df, all_embeddings
    
    # Standard handling for non-predefined splits
    else:
        # Determine which splits to load
        if split == 'all':
            splits_to_load = ['train', 'test', 'devel']
        else:
            splits_to_load = [split]
        
        all_metadata = []
        all_embeddings = {}
        
        for current_split in splits_to_load:
            split_dir = os.path.join(model_dir, current_split)
            
            # If split directory doesn't exist, try loading from model_dir directly
            if not os.path.exists(split_dir):
                if current_split == 'train' and os.path.exists(os.path.join(model_dir, 'embedding_metadata.csv')):
                    split_dir = model_dir
                else:
                    logger.warning(f"Split directory not found: {split_dir}")
                    continue
            
            # Load metadata
            metadata_path = os.path.join(split_dir, 'embedding_metadata.csv')
            if not os.path.exists(metadata_path):
                logger.warning(f"Metadata file not found: {metadata_path}")
                continue
            
            try:
                metadata_df = pd.read_csv(metadata_path)
                metadata_df['split'] = current_split
                all_metadata.append(metadata_df)
                logger.info(f"Loaded metadata for {len(metadata_df)} files from {current_split} split")
            except Exception as e:
                logger.error(f"Error loading metadata from {metadata_path}: {e}")
                continue
            
            # Find embedding files
            embedding_files = [f for f in os.listdir(split_dir) if f.endswith('_embeddings.npy')]
            layer_names = [os.path.splitext(f)[0].replace('_embeddings', '') for f in embedding_files]
            
            logger.info(f"Found embeddings for {len(layer_names)} layers in {current_split} split: {layer_names}")
            
            # Load embeddings for each layer
            for layer, filename in zip(layer_names, embedding_files):
                try:
                    embedding_array = np.load(os.path.join(split_dir, filename))
                    
                    if layer not in all_embeddings:
                        all_embeddings[layer] = []
                    
                    all_embeddings[layer].append(embedding_array)
                    logger.info(f"Loaded {layer} embeddings for {current_split} split with shape {embedding_array.shape}")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")
        
        # Combine metadata and embeddings
        if all_metadata:
            combined_metadata = pd.concat(all_metadata, ignore_index=True)
            logger.info(f"Combined metadata with {len(combined_metadata)} entries")
            
            # Combine embeddings
            for layer in all_embeddings:
                if len(all_embeddings[layer]) > 1:
                    all_embeddings[layer] = np.vstack(all_embeddings[layer])
                else:
                    all_embeddings[layer] = all_embeddings[layer][0]
                logger.info(f"Combined {layer} embeddings with final shape {all_embeddings[layer].shape}")
            
            return combined_metadata, all_embeddings
        else:
            logger.error("No metadata could be loaded")
            return None, {}

def check_data_quality(metadata_df, results_dir):
    """Check data quality and label distribution"""
    if metadata_df is None:
        logger.error("No metadata available")
        return
    
    # Check for missing values
    missing_values = metadata_df.isnull().sum()
    logger.info("Missing values in metadata:")
    for col, count in missing_values.items():
        if count > 0:
            logger.info(f"  {col}: {count}")
    
    # Display label distribution if available
    if 'label' in metadata_df.columns:
        logger.info("Label distribution:")
        label_counts = metadata_df['label'].value_counts()
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count}")
    
        # Visualize label distribution
        plt.figure(figsize=(10, 6))
        label_counts.plot(kind='bar')
        plt.title('Distribution of Stuttering Labels')
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'label_distribution.png'))
        plt.close()
        logger.info(f"Saved label distribution plot to {os.path.join(results_dir, 'label_distribution.png')}")
    else:
        logger.warning("No label column found in metadata")

def analyze_class_distribution(y, split_name=""):
    """Analyze and visualize class distribution"""
    logger.info(f"\n=== Class Distribution Analysis {split_name} ===")
    
    # Count classes
    unique_classes, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique_classes, counts))
    
    # Calculate imbalance ratio
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count
    
    logger.info(f"Number of classes: {len(unique_classes)}")
    logger.info(f"Total samples: {len(y)}")
    logger.info(f"Imbalance ratio: {imbalance_ratio:.2f}")
    
    # Log each class
    for class_name, count in class_dist.items():
        percentage = (count / len(y)) * 100
        logger.info(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    return class_dist, imbalance_ratio

def apply_smote_oversampling(X, y, k_neighbors=3, random_state=42):
    """Apply SMOTE oversampling to balance classes"""
    logger.info("\n=== Applying SMOTE Oversampling ===")
    
    # Original distribution
    original_dist = Counter(y)
    logger.info(f"Original distribution: {dict(original_dist)}")
    
    # Apply SMOTE
    # Use smaller k_neighbors for classes with few samples
    min_class_size = min(original_dist.values())
    actual_k_neighbors = min(k_neighbors, min_class_size - 1)
    
    if actual_k_neighbors < 1:
        logger.warning("Some classes have too few samples for SMOTE. Skipping oversampling.")
        return X, y
    
    smote = SMOTE(random_state=random_state, k_neighbors=actual_k_neighbors)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # New distribution
        new_dist = Counter(y_resampled)
        logger.info(f"After SMOTE distribution: {dict(new_dist)}")
        logger.info(f"Total samples: {len(y)} → {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        logger.info("Continuing without SMOTE...")
        return X, y

def compute_balanced_class_weights(y):
    """Compute balanced class weights"""
    logger.info("\n=== Computing Class Weights ===")
    
    # Get unique classes and compute weights
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    weight_dict = dict(zip(classes, class_weights))
    
    logger.info("Computed class weights:")
    for class_name, weight in weight_dict.items():
        logger.info(f"  {class_name}: {weight:.3f}")
    
    return weight_dict

def prepare_data(metadata_df, embeddings, layer_name):
    """Prepare data for model training"""
    if metadata_df is None or layer_name not in embeddings:
        logger.error(f"Data or {layer_name} embeddings not available")
        return None, None, None
    
    # Check if we have labels
    if 'label' not in metadata_df.columns:
        logger.error("No label column found in metadata")
        return None, None, None
    
    # Drop rows with missing labels
    valid_indices = metadata_df['label'].notna()
    if valid_indices.sum() == 0:
        logger.error("No valid labels found")
        return None, None, None
    
    # Check dimensions before indexing
    if sum(valid_indices) > embeddings[layer_name].shape[0]:
        logger.error(f"Dimension mismatch: metadata has {sum(valid_indices)} valid entries, but embeddings has {embeddings[layer_name].shape[0]} entries")
        return None, None, None
    
    try:
        # Get features and labels
        X = embeddings[layer_name][valid_indices]
        y = metadata_df.loc[valid_indices, 'label'].values
        
        logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        
        # Create label mapping for numeric labels if needed
        unique_labels = np.unique(y)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        idx_to_label = {i: label for label, i in label_to_idx.items()}
        
        return X, y, (label_to_idx, idx_to_label)
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None

def train_improved_models(X_train, y_train, X_test, y_test, use_smote=True, 
                         use_class_weights=True, smote_k_neighbors=3, random_state=42):
    """Train models with class balancing techniques"""
    
    results = []
    
    # Analyze original class distribution
    train_dist, train_imbalance = analyze_class_distribution(y_train, "Training")
    test_dist, test_imbalance = analyze_class_distribution(y_test, "Test")
    
    # Compute class weights
    class_weights = None
    if use_class_weights:
        class_weights = compute_balanced_class_weights(y_train)
    
    # Prepare data variations
    data_variants = [
        ("Original", X_train, y_train),
    ]
    
    if use_smote:
        X_train_smote, y_train_smote = apply_smote_oversampling(
            X_train, y_train, k_neighbors=smote_k_neighbors, random_state=random_state
        )
        data_variants.append(("SMOTE", X_train_smote, y_train_smote))
    
    # Model configurations
    model_configs = [
        ("SVM_Basic", SVC(kernel='rbf', C=10, probability=True, random_state=random_state)),
        ("SVM_Weighted", SVC(kernel='rbf', C=10, probability=True, random_state=random_state, 
                            class_weight='balanced' if use_class_weights else None)),
        ("RF_Basic", RandomForestClassifier(n_estimators=100, random_state=random_state)),
        ("RF_Weighted", RandomForestClassifier(n_estimators=100, random_state=random_state,
                                             class_weight='balanced' if use_class_weights else None)),
    ]
    
    # Train and evaluate all combinations
    for data_name, X_tr, y_tr in data_variants:
        for model_name, model in model_configs:
            
            # Skip weighted models on SMOTE data (redundant)
            if data_name == "SMOTE" and "Weighted" in model_name:
                continue
                
            logger.info(f"\n--- Training {model_name} on {data_name} data ---")
            
            # Create pipeline with scaling
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Train model
            pipeline.fit(X_tr, y_tr)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            f1_macro = f1_score(y_test, y_pred, average='macro')
            
            # Per-class metrics
            precision, recall, f1, support = precision_recall_fscore_support(
                y_test, y_pred, average=None, labels=np.unique(y_test)
            )
            
            # Store results
            result = {
                'Data': data_name,
                'Model': model_name,
                'Accuracy': accuracy,
                'Balanced_Accuracy': balanced_acc,
                'F1_Weighted': f1_weighted,
                'F1_Macro': f1_macro,
                'Pipeline': pipeline,
                'Predictions': y_pred
            }
            
            # Add per-class metrics
            for i, class_name in enumerate(np.unique(y_test)):
                result[f'{class_name}_Precision'] = precision[i]
                result[f'{class_name}_Recall'] = recall[i]
                result[f'{class_name}_F1'] = f1[i]
            
            results.append(result)
            
            # Log key metrics
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Balanced Accuracy: {balanced_acc:.4f}")
            logger.info(f"F1 (Weighted): {f1_weighted:.4f}")
            logger.info(f"F1 (Macro): {f1_macro:.4f}")
    
    return results

def train_svm_model(X, y, n_splits=5, random_state=42):
    """Train an SVM model with cross-validation (legacy function for compatibility)"""
    if X is None or y is None:
        logger.error("No valid data provided")
        return None, None
    
    # Create CV splits
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Define parameter grid
    param_grid = {
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf', probability=True, random_state=random_state, class_weight='balanced'))
    ])
    
    # Create grid search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='balanced_accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit grid search
    logger.info("Starting SVM grid search...")
    grid_search.fit(X, y)
    
    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    
    return grid_search.best_estimator_, cv_results

def evaluate_layers_and_classifiers(metadata_df, embeddings, results_dir, test_size=0.2, random_state=42):
    """Evaluate different layers and classifiers to find the best combination"""
    layer_names = list(embeddings.keys())
    if not layer_names or metadata_df is None:
        logger.error("No data available for evaluation")
        return None
    
    # Classifiers to evaluate
    classifiers = {
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, probability=True, random_state=random_state, class_weight='balanced'))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'))
        ])
    }
    
    # Results storage
    results = []
    
    # Evaluate each layer and classifier
    for layer_name in layer_names:
        # Prepare data
        X, y, _ = prepare_data(metadata_df, embeddings, layer_name)
        
        if X is None or y is None:
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        for clf_name, pipeline in classifiers.items():
            logger.info(f"Training {clf_name} on {layer_name}...")
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            balanced_acc = balanced_accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Store results
            results.append({
                'Layer': layer_name,
                'Classifier': clf_name,
                'Accuracy': accuracy,
                'Balanced_Accuracy': balanced_acc,
                'F1 Score': f1
            })
            
            logger.info(f"{clf_name} on {layer_name}: Accuracy={accuracy:.4f}, Balanced_Acc={balanced_acc:.4f}, F1={f1:.4f}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Visualize results
    plt.figure(figsize=(15, 12))
    
    metrics = ['Accuracy', 'Balanced_Accuracy', 'F1 Score']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(3, 1, i)
        for clf_name in classifiers.keys():
            subset = results_df[results_df['Classifier'] == clf_name]
            plt.plot(subset['Layer'], subset[metric], marker='o', label=clf_name, linewidth=2, markersize=8)
        plt.title(f'{metric} by Layer and Classifier', fontsize=14)
        plt.xlabel('Layer', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'layer_classifier_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved layer/classifier comparison plot to {os.path.join(results_dir, 'layer_classifier_comparison.png')}")
    
    return results_df

def create_comparison_visualizations(results, results_dir, y_test, layer_name):
    """Create comprehensive comparison visualizations"""
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # 1. Performance comparison heatmap
    plt.figure(figsize=(15, 10))
    
    # Create pivot table for metrics
    metrics = ['Accuracy', 'Balanced_Accuracy', 'F1_Weighted', 'F1_Macro']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        
        pivot_data = results_df.pivot(index='Data', columns='Model', values=metric)
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='Blues', 
                   cbar_kws={'label': metric})
        plt.title(f'{metric} Comparison - {layer_name}')
        plt.tight_layout()
    
    plt.savefig(os.path.join(results_dir, 'performance_comparison_heatmap.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved performance comparison heatmap")
    
    # 2. Best model identification
    best_result = max(results, key=lambda x: x['Balanced_Accuracy'])
    
    logger.info(f"Best model: {best_result['Model']} on {best_result['Data']} data")
    logger.info(f"Best Balanced Accuracy: {best_result['Balanced_Accuracy']:.4f}")
    
    return results_df, best_result

def save_detailed_results(results_df, best_result, results_dir, model_type, layer_name):
    """Save detailed results to files"""
    
    # Save results DataFrame
    results_df.to_csv(os.path.join(results_dir, 'all_results_comparison.csv'), index=False)
    
    # Save best model summary
    with open(os.path.join(results_dir, 'best_model_summary.txt'), 'w') as f:
        f.write("=== Best Model Results ===\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Best Configuration: {best_result['Model']} on {best_result['Data']} data\n\n")
        f.write(f"Metrics:\n")
        f.write(f"  Accuracy: {best_result['Accuracy']:.4f}\n")
        f.write(f"  Balanced Accuracy: {best_result['Balanced_Accuracy']:.4f}\n")
        f.write(f"  F1 (Weighted): {best_result['F1_Weighted']:.4f}\n")
        f.write(f"  F1 (Macro): {best_result['F1_Macro']:.4f}\n\n")
        
        # Per-class performance
        f.write("Per-Class Performance:\n")
        for key, value in best_result.items():
            if '_Recall' in key:
                class_name = key.replace('_Recall', '')
                precision_key = f'{class_name}_Precision'
                f1_key = f'{class_name}_F1'
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {best_result.get(precision_key, 'N/A'):.4f}\n")
                f.write(f"    Recall: {value:.4f}\n")
                f.write(f"    F1: {best_result.get(f1_key, 'N/A'):.4f}\n")
    
    logger.info(f"Saved detailed results to {results_dir}")

def identify_best_model(eval_results):
    """Identify the best layer and classifier based on balanced accuracy"""
    if eval_results is None or len(eval_results) == 0:
        logger.error("No evaluation results available")
        return None, None
    
    # Find best combination based on Balanced Accuracy
    best_row = eval_results.loc[eval_results['Balanced_Accuracy'].idxmax()]
    best_layer = best_row['Layer']
    best_classifier = best_row['Classifier']
    
    logger.info(f"Best model: {best_classifier} on {best_layer}")
    logger.info(f"Best Balanced Accuracy: {best_row['Balanced_Accuracy']:.4f}")
    logger.info(f"Best Regular Accuracy: {best_row['Accuracy']:.4f}")
    logger.info(f"Best F1 Score: {best_row['F1 Score']:.4f}")
    
    return best_layer, best_classifier

def evaluate_best_model(metadata_df, embeddings, best_layer, best_classifier, results_dir, test_size=0.2, random_state=42):
    """Perform detailed evaluation of the best model with normalized confusion matrix"""
    if best_layer is None or best_classifier is None:
        logger.error("No best model identified")
        return None
    
    # Prepare data
    X, y, label_mapping = prepare_data(metadata_df, embeddings, best_layer)
    
    if X is None or y is None:
        logger.error("Could not prepare data")
        return None
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Create classifier
    if best_classifier == 'SVM':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10, probability=True, random_state=random_state, class_weight='balanced'))
        ])
    elif best_classifier == 'Random Forest':
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced'))
        ])
    else:
        logger.error(f"Unknown classifier: {best_classifier}")
        return None
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_pred = pipeline.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    logger.info("Classification Report:")
    logger.info("\n" + report)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create normalized confusion matrix (row normalization - each row sums to 1)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot regular confusion matrix
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 1, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix: {best_classifier} on {best_layer}')
    plt.colorbar()
    
    # Add labels
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    # Plot normalized confusion matrix
    plt.subplot(2, 1, 2)
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Normalized Confusion Matrix: {best_classifier} on {best_layer}')
    plt.colorbar()
    
    # Add labels
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations (percentages)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            # Format as percentage with 1 decimal place
            plt.text(j, i, format(cm_normalized[i, j]*100, '.1f') + '%',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()
    logger.info(f"Saved confusion matrices to {os.path.join(results_dir, 'confusion_matrix.png')}")
    
    # Save separate normalized confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Normalized Confusion Matrix: {best_classifier} on {best_layer}')
    plt.colorbar()
    
    # Add labels
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add text annotations (percentages)
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            plt.text(j, i, format(cm_normalized[i, j]*100, '.1f') + '%',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > 0.5 else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'normalized_confusion_matrix.png'))
    plt.close()
    logger.info(f"Saved normalized confusion matrix to {os.path.join(results_dir, 'normalized_confusion_matrix.png')}")
    
    # Save detailed results to file
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Best model: {best_classifier} on {best_layer}\n\n")
        f.write(report)
    
    return pipeline

def save_best_model(model, layer_name, model_type, model_config="", results_dir="models"):
    """Save the best model and related information"""
    if model is None or layer_name is None:
        logger.error("No model to save")
        return
    
    # Create models directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Save model
    try:
        import joblib
        model_path = os.path.join(results_dir, f'best_stuttering_model_{model_type}_{layer_name}_{model_config}.joblib')
        joblib.dump(model, model_path)
        
        # Save layer information
        info = {
            'model_type': model_type,
            'layer_name': layer_name,
            'model_config': model_config,
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save as JSON
        import json
        with open(os.path.join(results_dir, f'model_info_{model_type}_{layer_name}_{model_config}.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        logger.info(f"Model and info saved to {results_dir}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Load data
    logger.info(f"Loading {args.model_type} embeddings from {args.embeddings_dir}")
    metadata_df, embeddings = load_data(args.embeddings_dir, args.model_type, args.split)

    if metadata_df is None or not embeddings:
        logger.error("Failed to load data. Exiting.")
        return

    # Check data quality
    check_data_quality(metadata_df, args.results_dir)

    # === IF using predefined splits (train/test/devel), respect them ===
    if args.split == 'predefined':
        logger.info("Using predefined splits: training on 'train', evaluating on 'test' and 'devel'")

        # Separate metadata
        train_meta = metadata_df[metadata_df['split'] == 'train'].reset_index(drop=True)
        test_meta = metadata_df[metadata_df['split'].isin(['test', 'devel'])].reset_index(drop=True)
        
        logger.info(f"Train split has {len(train_meta)} samples")
        logger.info(f"Test split has {len(test_meta)} samples")

        best_model = None
        best_layer = None
        best_balanced_acc = 0.0
        all_layer_results = {}

        for layer_name in embeddings.keys():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Layer: {layer_name}")
            logger.info(f"{'='*60}")
            
            # Filter embeddings for train split
            n_train = len(train_meta)
            train_embeddings = {layer_name: embeddings[layer_name][:n_train]}
            
            # Filter embeddings for test split
            n_test = len(test_meta)
            test_embeddings = {layer_name: embeddings[layer_name][n_train:n_train+n_test]}
            
            logger.info(f"Filtered {layer_name} embeddings to {n_train} train samples and {n_test} test samples")
            
            # Prepare data with filtered embeddings
            X_train, y_train, _ = prepare_data(train_meta, train_embeddings, layer_name)
            X_test, y_test, _ = prepare_data(test_meta, test_embeddings, layer_name)
            
            if X_train is None or X_test is None:
                logger.warning(f"Skipping layer {layer_name} due to data preparation issues")
                continue

            # Create layer-specific results directory
            layer_results_dir = os.path.join(args.results_dir, f'layer_{layer_name}')
            os.makedirs(layer_results_dir, exist_ok=True)
            
            # Train models with class balancing
            results = train_improved_models(
                X_train, y_train, X_test, y_test,
                use_smote=args.use_smote,
                use_class_weights=args.use_class_weights,
                smote_k_neighbors=args.smote_k_neighbors
            )
            
            # Visualizations and analysis
            results_df, best_result = create_comparison_visualizations(results, layer_results_dir, y_test, layer_name)
            
            # Create confusion matrix for best result
            best_pipeline = best_result['Pipeline']
            y_pred_best = best_result['Predictions']
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred_best)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                       xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
            plt.title(f'Best Model Confusion Matrix: {best_result["Model"]} on {best_result["Data"]} data\n'
                     f'Layer {layer_name} - Balanced Accuracy: {best_result["Balanced_Accuracy"]:.3f}')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(os.path.join(layer_results_dir, 'best_confusion_matrix.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save detailed results
            save_detailed_results(results_df, best_result, layer_results_dir, 
                                args.model_type, layer_name)
            
            # Track best overall result
            if best_result['Balanced_Accuracy'] > best_balanced_acc:
                best_balanced_acc = best_result['Balanced_Accuracy']
                best_model = best_result
                best_layer = layer_name
            
            # Store results for this layer
            all_layer_results[layer_name] = results_df
            
            # Classification report
            report = classification_report(y_test, y_pred_best)
            logger.info(f"\nClassification Report for Best Model ({layer_name}):")
            logger.info(f"\n{report}")
            
            # Save classification report
            with open(os.path.join(layer_results_dir, 'classification_report.txt'), 'w') as f:
                f.write(f"Best model: {best_result['Model']} on {best_result['Data']} data\n")
                f.write(f"Layer: {layer_name}\n")
                f.write(f"Balanced Accuracy: {best_result['Balanced_Accuracy']:.4f}\n")
                f.write(f"Regular Accuracy: {best_result['Accuracy']:.4f}\n")
                f.write(f"F1 Weighted: {best_result['F1_Weighted']:.4f}\n")
                f.write(f"F1 Macro: {best_result['F1_Macro']:.4f}\n\n")
                f.write(report)
        
        # Create overall comparison across layers
        if all_layer_results:
            logger.info(f"\n{'='*80}")
            logger.info(f"CREATING OVERALL LAYER COMPARISON")
            logger.info(f"{'='*80}")
            
            # Combine best results from each layer
            layer_comparison = []
            for layer_name, results_df in all_layer_results.items():
                best_for_layer = results_df.loc[results_df['Balanced_Accuracy'].idxmax()]
                layer_comparison.append({
                    'Layer': layer_name,
                    'Best_Model': best_for_layer['Model'],
                    'Best_Data': best_for_layer['Data'],
                    'Accuracy': best_for_layer['Accuracy'],
                    'Balanced_Accuracy': best_for_layer['Balanced_Accuracy'],
                    'F1_Weighted': best_for_layer['F1_Weighted'],
                    'F1_Macro': best_for_layer['F1_Macro']
                })
            
            layer_comparison_df = pd.DataFrame(layer_comparison)
            
            # Save layer comparison
            layer_comparison_df.to_csv(os.path.join(args.results_dir, 'layer_comparison_summary.csv'), index=False)
            
            # Visualize layer comparison
            plt.figure(figsize=(15, 10))
            
            metrics = ['Accuracy', 'Balanced_Accuracy', 'F1_Weighted', 'F1_Macro']
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(2, 2, i)
                plt.plot(layer_comparison_df['Layer'], layer_comparison_df[metric], 
                        marker='o', linewidth=2, markersize=8)
                plt.title(f'{metric} by Layer', fontsize=14)
                plt.xlabel('Layer', fontsize=12)
                plt.ylabel(metric, fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                
                # Highlight best layer
                best_idx = layer_comparison_df[metric].idxmax()
                best_layer_name = layer_comparison_df.iloc[best_idx]['Layer']
                best_value = layer_comparison_df.iloc[best_idx][metric]
                plt.annotate(f'Best: {best_layer_name}\n{best_value:.3f}', 
                           xy=(best_idx, best_value), xytext=(10, 10),
                           textcoords='offset points', ha='left',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, 'overall_layer_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved overall layer comparison to {args.results_dir}")

    else:
        # Default: combine all and perform random split
        eval_results = evaluate_layers_and_classifiers(
            metadata_df, embeddings, args.results_dir, test_size=args.test_size
        )

        if eval_results is None:
            logger.error("Evaluation failed. Exiting.")
            return

        best_layer, best_classifier = identify_best_model(eval_results)

        if best_layer is None or best_classifier is None:
            logger.error("Failed to identify best model. Exiting.")
            return

        best_model = evaluate_best_model(
            metadata_df, embeddings, best_layer, best_classifier, args.results_dir, test_size=args.test_size
        )

        if best_model is None:
            logger.error("Detailed evaluation failed. Exiting.")
            return

        save_best_model(best_model, best_layer, args.model_type)

    # Final summary
    if args.split == 'predefined' and best_model:
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST OVERALL RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Best Layer: {best_layer}")
        logger.info(f"Best Configuration: {best_model['Model']} on {best_model['Data']} data")
        logger.info(f"Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}")
        logger.info(f"Regular Accuracy: {best_model['Accuracy']:.4f}")
        logger.info(f"F1 (Weighted): {best_model['F1_Weighted']:.4f}")
        logger.info(f"F1 (Macro): {best_model['F1_Macro']:.4f}")
        
        # Save best model
        model_config = f"{best_model['Model']}_{best_model['Data']}"
        model_path = save_best_model(best_model['Pipeline'], best_layer, args.model_type, 
                                   model_config, os.path.join(args.results_dir, 'models'))
        
        # Save final summary
        with open(os.path.join(args.results_dir, 'final_summary.txt'), 'w') as f:
            f.write("=== FINAL EXPERIMENT SUMMARY ===\n\n")
            f.write(f"Dataset: {args.model_type} embeddings\n")
            f.write(f"Split strategy: {args.split}\n")
            f.write(f"SMOTE used: {args.use_smote}\n")
            f.write(f"Class weights used: {args.use_class_weights}\n\n")
            f.write(f"Best overall configuration:\n")
            f.write(f"  Layer: {best_layer}\n")
            f.write(f"  Model: {best_model['Model']}\n")
            f.write(f"  Data variant: {best_model['Data']}\n")
            f.write(f"  Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}\n")
            f.write(f"  Regular Accuracy: {best_model['Accuracy']:.4f}\n")
            f.write(f"  F1 Weighted: {best_model['F1_Weighted']:.4f}\n")
            f.write(f"  F1 Macro: {best_model['F1_Macro']:.4f}\n\n")
            f.write(f"Model saved to: {model_path}\n")
            
        logger.info(f"Saved final summary to {args.results_dir}")
        
    elif args.split != 'predefined':
        # Summary for random split approach
        logger.info("\n=== Model Training and Evaluation Summary ===")
        logger.info(f"Model type: {args.model_type}")
        logger.info(f"Evaluated {len(embeddings)} layers with multiple classifiers")
        logger.info(f"Best model: {best_classifier} on {best_layer}")
        if 'eval_results' in locals():
            best_row = eval_results[(eval_results['Layer'] == best_layer) & (eval_results['Classifier'] == best_classifier)].iloc[0]
            logger.info(f"Balanced Accuracy: {best_row['Balanced_Accuracy']:.4f}")
            logger.info(f"F1 Score: {best_row['F1 Score']:.4f}")
            logger.info(f"Accuracy: {best_row['Accuracy']:.4f}")
        logger.info("\nResults and models saved to disk.")

if __name__ == "__main__":
    main()