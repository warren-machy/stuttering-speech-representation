#!/usr/bin/env python3
# model_training_01.py - Train stuttering classification models with balanced approach

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
import torch
import torchaudio
import random
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
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

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
    parser = argparse.ArgumentParser(description='Train stuttering classification models with balanced approach')
    
    parser.add_argument('--embeddings_dir', type=str, required=True,
                        help='Directory with extracted embeddings')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory to save results')
    parser.add_argument('--model_type', type=str, default='wavlm',
                        choices=['whisper', 'wavlm', 'wavlm_large', 'bestrq', 'combined', 'whisper_large_fixed'],
                        help='Type of embeddings to use')
    parser.add_argument('--split', type=str, default='predefined',
                        choices=['train_test', 'predefined', 'all'],
                        help='How to split the data (predefined=use dataset splits, train_test=manual split)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test size for train_test split')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                        help='How many augmented samples to create per minority sample')
    parser.add_argument('--minority_threshold', type=int, default=100,
                        help='Classes with fewer samples than this will be augmented')
    parser.add_argument('--smote_k_neighbors', type=int, default=3,
                        help='Number of neighbors for SMOTE')
    parser.add_argument('--model_name', type=str, default='microsoft/wavlm-large',
                        help='Model name for re-extracting embeddings from augmented audio')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for model (cuda:0, cuda:1, cpu)')
    parser.add_argument('--classifier', type=str, default='svm',
                        choices=['svm', 'rf', 'xgb', 'all'],
                        help='Which classifier to use (svm, rf, xgb, or all)')
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
    
    else:
        logger.error("This script currently supports only predefined splits for augmentation")
        return None, {}

def augment_audio(waveform, sample_rate=16000, augmentation_type='random'):
    """Apply audio augmentation techniques suitable for stuttering data"""
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    # Ensure proper shape [1, samples] for mono audio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Ensure tensor doesn't require gradients
    waveform = waveform.detach()
    
    # Random augmentation selection
    if augmentation_type == 'random':
        augmentation_type = random.choice(['speed', 'noise', 'pitch', 'volume'])
    
    try:
        if augmentation_type == 'speed':
            # Speed perturbation (0.9x to 1.1x speed)
            speed_factor = random.uniform(0.9, 1.1)
            new_sample_rate = int(sample_rate * speed_factor)
            resampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate)
            waveform_resampled = resampler(waveform)
            resampler_back = torchaudio.transforms.Resample(new_sample_rate, sample_rate)
            waveform = resampler_back(waveform_resampled)
            
        elif augmentation_type == 'noise':
            # Add Gaussian noise (low level to preserve stuttering characteristics)
            noise_factor = random.uniform(0.005, 0.02)
            noise = torch.randn_like(waveform) * noise_factor
            waveform = waveform + noise
            
        elif augmentation_type == 'pitch':
            # Pitch shifting (±2 semitones)
            n_steps = random.randint(-2, 2)
            if n_steps != 0:
                pitch_shift = torchaudio.transforms.PitchShift(sample_rate, n_steps=n_steps)
                waveform = pitch_shift(waveform)
                
        elif augmentation_type == 'volume':
            # Volume perturbation (0.8x to 1.2x volume)
            volume_factor = random.uniform(0.8, 1.2)
            waveform = waveform * volume_factor
            
        # Normalize to prevent clipping
        waveform = torch.clamp(waveform, -1.0, 1.0)
        
        # Detach from computation graph before converting to numpy
        return waveform.detach().squeeze().numpy()
        
    except Exception as e:
        logger.warning(f"Augmentation failed: {e}. Returning original audio.")
        return waveform.detach().squeeze().numpy()

def load_audio_for_augmentation(file_path, target_sr=16000):
    """Load audio file for augmentation"""
    try:
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)
        
        # Detach from computation graph and convert to numpy
        return waveform.detach().squeeze().numpy()
    except Exception as e:
        logger.error(f"Error loading audio {file_path}: {e}")
        return None

def extract_embeddings_from_audio_wavlm(audio_array, model, feature_extractor, device, layer_indices):
    """Extract WavLM embeddings from augmented audio"""
    try:
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(
                inputs.input_values,
                output_hidden_states=True,
                return_dict=True
            )
        
        hidden_states = outputs.hidden_states
        embeddings = {}
        
        for idx in layer_indices:
            if idx < len(hidden_states):
                hidden_state = hidden_states[idx]
                embedding = torch.mean(hidden_state, dim=1).cpu().numpy()
                embeddings[f"layer_{idx}"] = embedding.flatten()
        
        return embeddings
    except Exception as e:
        logger.error(f"Error extracting WavLM embeddings: {e}")
        return None

def extract_embeddings_from_audio_whisper(audio_array, model, processor, device, layer_names):
    """Extract Whisper embeddings from augmented audio"""
    try:
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True
            )
            
            decoder_outputs = model.decoder(
                input_ids=torch.zeros((1, 1), dtype=torch.long).to(device),
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                output_hidden_states=True,
                return_dict=True
            )
        
        encoder_states = encoder_outputs.hidden_states
        decoder_states = decoder_outputs.hidden_states
        embeddings = {}
        
        for layer_name in layer_names:
            if layer_name.startswith('encoder_layer_'):
                idx = int(layer_name.split('_')[-1])
                if idx < len(encoder_states):
                    hidden_state = encoder_states[idx]
                    embedding = torch.mean(hidden_state, dim=1).cpu().numpy()
                    embeddings[layer_name] = embedding.flatten()
            elif layer_name.startswith('decoder_layer_'):
                idx = int(layer_name.split('_')[-1])
                if idx < len(decoder_states):
                    hidden_state = decoder_states[idx]
                    embedding = hidden_state.squeeze(1).cpu().numpy()
                    embeddings[layer_name] = embedding.flatten()
        
        return embeddings
    except Exception as e:
        logger.error(f"Error extracting Whisper embeddings: {e}")
        return None

def apply_data_augmentation(train_meta, train_embeddings, model, feature_extractor, device, 
                           layer_names, model_type, augmentation_factor=3, minority_threshold=100):
    """Apply data augmentation to minority classes"""
    logger.info("\n=== Applying Data Augmentation ===")
    
    if 'path' not in train_meta.columns:
        logger.warning("No audio file paths found. Skipping data augmentation.")
        return train_meta, train_embeddings
    
    if 'label' not in train_meta.columns:
        logger.warning("No labels found. Skipping data augmentation.")
        return train_meta, train_embeddings
    
    class_counts = train_meta['label'].value_counts()
    minority_classes = class_counts[class_counts < minority_threshold].index.tolist()
    
    logger.info(f"Classes to augment (< {minority_threshold} samples): {minority_classes}")
    logger.info(f"Augmentation factor: {augmentation_factor}")
    
    if not minority_classes:
        logger.info("No minority classes found. Skipping augmentation.")
        return train_meta, train_embeddings
    
    augmented_metadata = []
    augmented_embeddings = {layer: [] for layer in train_embeddings.keys()}
    
    for class_name in minority_classes:
        class_samples = train_meta[train_meta['label'] == class_name]
        logger.info(f"Augmenting {len(class_samples)} samples for class '{class_name}'")
        
        for idx, row in class_samples.iterrows():
            audio_path = row['path']
            original_audio = load_audio_for_augmentation(audio_path)
            
            if original_audio is None:
                continue
            
            for aug_idx in range(augmentation_factor):
                try:
                    augmented_audio = augment_audio(original_audio, augmentation_type='random')
                    
                    if model_type.lower() in ['wavlm', 'wavlm_large']:
                        layer_indices = [int(name.split('_')[1]) for name in layer_names if name.startswith('layer_')]
                        aug_embeddings = extract_embeddings_from_audio_wavlm(
                            augmented_audio, model, feature_extractor, device, layer_indices
                        )
                    elif model_type.lower() in ['whisper', 'whisper_large_fixed']:
                        aug_embeddings = extract_embeddings_from_audio_whisper(
                            augmented_audio, model, feature_extractor, device, layer_names
                        )
                    else:
                        logger.warning(f"Unsupported model type for augmentation: {model_type}")
                        continue
                    
                    if aug_embeddings is None:
                        continue
                    
                    aug_metadata = row.copy()
                    aug_metadata['filename'] = f"{row['filename']}_aug_{aug_idx}"
                    aug_metadata['augmented'] = True
                    aug_metadata['augmentation_type'] = 'mixed'
                    
                    augmented_metadata.append(aug_metadata)
                    
                    for layer_name, embedding in aug_embeddings.items():
                        if layer_name in augmented_embeddings:
                            augmented_embeddings[layer_name].append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to augment sample {row['filename']}: {e}")
                    continue
    
    if augmented_metadata:
        for layer_name in augmented_embeddings:
            if augmented_embeddings[layer_name]:
                augmented_embeddings[layer_name] = np.array(augmented_embeddings[layer_name])
        
        aug_meta_df = pd.DataFrame(augmented_metadata)
        combined_meta = pd.concat([train_meta, aug_meta_df], ignore_index=True)
        
        combined_embeddings = {}
        for layer_name in train_embeddings:
            original = train_embeddings[layer_name]
            if layer_name in augmented_embeddings and len(augmented_embeddings[layer_name]) > 0:
                augmented = augmented_embeddings[layer_name]
                combined_embeddings[layer_name] = np.vstack([original, augmented])
            else:
                combined_embeddings[layer_name] = original
            
            logger.info(f"Combined {layer_name}: {original.shape[0]} original + "
                       f"{len(augmented_embeddings[layer_name]) if layer_name in augmented_embeddings else 0} "
                       f"augmented = {combined_embeddings[layer_name].shape[0]} total")
        
        logger.info(f"Data augmentation complete: {len(train_meta)} → {len(combined_meta)} samples")
        return combined_meta, combined_embeddings
    
    else:
        logger.warning("No augmented samples were created.")
        return train_meta, train_embeddings

def apply_smote_oversampling(X, y, k_neighbors=3, random_state=42):
    """Apply SMOTE oversampling to balance classes"""
    logger.info("\n=== Applying SMOTE Oversampling ===")
    
    original_dist = Counter(y)
    logger.info(f"Original distribution: {dict(original_dist)}")
    
    min_class_size = min(original_dist.values())
    actual_k_neighbors = min(k_neighbors, min_class_size - 1)
    
    if actual_k_neighbors < 1:
        logger.warning("Some classes have too few samples for SMOTE. Skipping oversampling.")
        return X, y
    
    smote = SMOTE(random_state=random_state, k_neighbors=actual_k_neighbors)
    
    try:
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        new_dist = Counter(y_resampled)
        logger.info(f"After SMOTE distribution: {dict(new_dist)}")
        logger.info(f"Total samples: {len(y)} → {len(y_resampled)}")
        
        return X_resampled, y_resampled
    
    except Exception as e:
        logger.error(f"SMOTE failed: {e}")
        logger.info("Continuing without SMOTE...")
        return X, y

def prepare_data(metadata_df, embeddings, layer_name):
    """Prepare data for model training"""
    if metadata_df is None or layer_name not in embeddings:
        logger.error(f"Data or {layer_name} embeddings not available")
        return None, None, None
    
    if 'label' not in metadata_df.columns:
        logger.error("No label column found in metadata")
        return None, None, None
    
    valid_indices = metadata_df['label'].notna()
    if valid_indices.sum() == 0:
        logger.error("No valid labels found")
        return None, None, None
    
    if sum(valid_indices) > embeddings[layer_name].shape[0]:
        logger.error(f"Dimension mismatch: metadata has {sum(valid_indices)} valid entries, but embeddings has {embeddings[layer_name].shape[0]} entries")
        return None, None, None
    
    try:
        X = embeddings[layer_name][valid_indices]
        y = metadata_df.loc[valid_indices, 'label'].values
        
        logger.info(f"Prepared data with {X.shape[0]} samples and {X.shape[1]} features")
        
        unique_labels = np.unique(y)
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        idx_to_label = {i: label for label, i in label_to_idx.items()}
        
        return X, y, (label_to_idx, idx_to_label)
    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        return None, None, None

def train_balanced_model(X_train, y_train, X_test, y_test, classifier_type='svm', smote_k_neighbors=3, random_state=42):
    """Train a single balanced model with SMOTE and class weights"""
    
    logger.info(f"\n=== Training Balanced {classifier_type.upper()} Model ===")
    
    # Analyze original class distribution
    original_dist = Counter(y_train)
    logger.info(f"Training distribution before SMOTE: {dict(original_dist)}")
    
    # Apply SMOTE
    X_train_balanced, y_train_balanced = apply_smote_oversampling(
        X_train, y_train, k_neighbors=smote_k_neighbors, random_state=random_state
    )
    
    # For XGBoost, we need to encode string labels to numeric
    if classifier_type.lower() == 'xgb':
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        
        # Fit on all possible labels (train + test)
        all_labels = np.concatenate([y_train_balanced, y_test])
        label_encoder.fit(all_labels)
        
        # Transform labels
        y_train_encoded = label_encoder.transform(y_train_balanced)
        y_test_encoded = label_encoder.transform(y_test)
        
        logger.info(f"Encoded {len(label_encoder.classes_)} classes for XGBoost: {label_encoder.classes_}")
    else:
        y_train_encoded = y_train_balanced
        y_test_encoded = y_test
        label_encoder = None
    
    # Create classifier with class weights
    if classifier_type.lower() == 'svm':
        classifier = SVC(kernel='rbf', C=10, probability=True, random_state=random_state, class_weight='balanced')
    elif classifier_type.lower() == 'rf':
        classifier = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    elif classifier_type.lower() == 'xgb':
        try:
            from xgboost import XGBClassifier
            # XGBoost handles class imbalance well internally
            classifier = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        except ImportError:
            logger.error("XGBoost not installed. Please install with: pip install xgboost")
            return None
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])
    
    # Train model
    logger.info(f"Training {classifier_type.upper()} with {len(X_train_balanced)} balanced samples...")
    pipeline.fit(X_train_balanced, y_train_encoded)
    
    # Evaluate
    y_pred_encoded = pipeline.predict(X_test)
    
    # For XGBoost, decode predictions back to original labels
    if classifier_type.lower() == 'xgb' and label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred_encoded)
        y_test_for_metrics = y_test  # Use original string labels for metrics
    else:
        y_pred = y_pred_encoded
        y_test_for_metrics = y_test
    
    # Calculate metrics (using standard ML terminology)
    balanced_accuracy = balanced_accuracy_score(y_test_for_metrics, y_pred)
    f1_weighted = f1_score(y_test_for_metrics, y_pred, average='weighted')
    f1_macro = f1_score(y_test_for_metrics, y_pred, average='macro')
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test_for_metrics, y_pred, average=None, labels=np.unique(y_test_for_metrics)
    )
    
    # Store results (balanced accuracy is the main metric for imbalanced datasets)
    result = {
        'Model': f"Balanced_{classifier_type.upper()}",
        'Balanced_Accuracy': balanced_accuracy,  # Standard term for imbalanced classification
        'F1_Weighted': f1_weighted,
        'F1_Macro': f1_macro,
        'Pipeline': pipeline,
        'Predictions': y_pred,
        'Label_Encoder': label_encoder  # Store for later use if needed
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(np.unique(y_test_for_metrics)):
        result[f'{class_name}_Precision'] = precision[i]
        result[f'{class_name}_Recall'] = recall[i]
        result[f'{class_name}_F1'] = f1[i]
    
    # Log key metrics (using standard terminology)
    logger.info(f"Balanced Accuracy: {balanced_accuracy:.4f}")
    logger.info(f"F1 (Weighted): {f1_weighted:.4f}")
    logger.info(f"F1 (Macro): {f1_macro:.4f}")
    
    return result

def create_visualizations(result, y_test, results_dir, layer_name, classifier_type):
    """Create visualizations for the model results"""
    
    y_pred = result['Predictions']
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    
    # Raw confusion matrix
    plt.subplot(2, 1, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Confusion Matrix: {result["Model"]} - {layer_name}\n'
             f'Balanced Accuracy: {result["Balanced_Accuracy"]:.3f}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Normalized confusion matrix
    plt.subplot(2, 1, 2)
    sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
               xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f'Normalized Confusion Matrix: {result["Model"]} - {layer_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'confusion_matrix_{classifier_type}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class performance bar chart
    classes = np.unique(y_test)
    metrics = ['Precision', 'Recall', 'F1']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(classes))
    width = 0.25
    
    for i, metric in enumerate(metrics):
        values = [result[f'{cls}_{metric}'] for cls in classes]
        ax.bar(x + i*width, values, width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title(f'Per-Class Performance: {result["Model"]} - {layer_name}')
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, f'per_class_performance_{classifier_type}.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved visualizations to {results_dir}")

def save_results(result, results_dir, model_type, layer_name, classifier_type):
    """Save detailed results to files"""
    
    # Save results summary
    with open(os.path.join(results_dir, f'results_summary_{classifier_type}.txt'), 'w') as f:
        f.write(f"=== {result['Model']} Results ===\n\n")
        f.write(f"Model Type: {model_type}\n")
        f.write(f"Layer: {layer_name}\n")
        f.write(f"Classifier: {classifier_type}\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"  Balanced Accuracy: {result['Balanced_Accuracy']:.4f}\n")
        f.write(f"  F1 (Weighted): {result['F1_Weighted']:.4f}\n")
        f.write(f"  F1 (Macro): {result['F1_Macro']:.4f}\n\n")
        f.write(f"Note: Balanced Accuracy is the primary metric for imbalanced datasets.\n")
        f.write(f"It measures the average recall across all classes.\n\n")
        
        # Per-class performance
        f.write("Per-Class Performance:\n")
        for key, value in result.items():
            if '_Recall' in key:
                class_name = key.replace('_Recall', '')
                precision_key = f'{class_name}_Precision'
                f1_key = f'{class_name}_F1'
                f.write(f"  {class_name}:\n")
                f.write(f"    Precision: {result.get(precision_key, 'N/A'):.4f}\n")
                f.write(f"    Recall: {value:.4f}\n")
                f.write(f"    F1: {result.get(f1_key, 'N/A'):.4f}\n")
    
    logger.info(f"Saved results summary to {results_dir}")

def save_model(model, layer_name, model_type, classifier_type, results_dir):
    """Save the trained model"""
    if model is None or layer_name is None:
        logger.error("No model to save")
        return
    
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    try:
        import joblib
        model_path = os.path.join(model_dir, f'balanced_{classifier_type}_{model_type}_{layer_name}.joblib')
        joblib.dump(model, model_path)
        
        # Save model information
        info = {
            'model_type': model_type,
            'layer_name': layer_name,
            'classifier_type': classifier_type,
            'approach': 'balanced_with_augmentation_and_smote',
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open(os.path.join(model_dir, f'model_info_{classifier_type}_{model_type}_{layer_name}.json'), 'w') as f:
            json.dump(info, f, indent=4)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        return None

def main():
    """Main execution function"""
    args = parse_args()

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)

    # Set up device for augmentation
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")

    # Load data
    logger.info(f"Loading {args.model_type} embeddings from {args.embeddings_dir}")
    metadata_df, embeddings = load_data(args.embeddings_dir, args.model_type, args.split)

    if metadata_df is None or not embeddings:
        logger.error("Failed to load data. Exiting.")
        return

    # Display label distribution
    if 'label' in metadata_df.columns:
        logger.info("Overall label distribution:")
        label_counts = metadata_df['label'].value_counts()
        for label, count in label_counts.items():
            logger.info(f"  {label}: {count}")

    # === PREDEFINED SPLITS WITH BALANCED APPROACH ===
    if args.split == 'predefined':
        logger.info("Using predefined splits with balanced approach (augmentation + SMOTE + class weights)")

        # Separate metadata
        train_meta = metadata_df[metadata_df['split'] == 'train'].reset_index(drop=True)
        test_meta = metadata_df[metadata_df['split'].isin(['test', 'devel'])].reset_index(drop=True)
        
        logger.info(f"Train split has {len(train_meta)} samples")
        logger.info(f"Test split has {len(test_meta)} samples")

        # Load model for augmentation
        model = None
        feature_extractor = None
        layer_names = []
        
        logger.info(f"Loading model for data augmentation: {args.model_name}")
        try:
            if args.model_type.lower() in ['wavlm', 'wavlm_large']:
                from transformers import Wav2Vec2FeatureExtractor, WavLMModel
                feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
                model = WavLMModel.from_pretrained(args.model_name).to(device)
                layer_names = [layer for layer in embeddings.keys() if layer.startswith('layer_')]
                logger.info(f"WavLM layer names for augmentation: {layer_names}")
                
            elif args.model_type.lower() in ['whisper', 'whisper_large_fixed']:
                from transformers import WhisperProcessor, WhisperModel
                feature_extractor = WhisperProcessor.from_pretrained(args.model_name)
                model = WhisperModel.from_pretrained(args.model_name).to(device)
                layer_names = [layer for layer in embeddings.keys() 
                             if layer.startswith('encoder_layer_') or layer.startswith('decoder_layer_')]
                logger.info(f"Whisper layer names for augmentation: {layer_names}")
                
            else:
                logger.warning(f"Data augmentation not supported for model type: {args.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to load model for augmentation: {e}")
            logger.warning("Continuing without data augmentation.")
            model = None

        best_model = None
        best_layer = None
        best_balanced_acc = 0.0
        all_layer_results = []

        # Determine which classifiers to use
        classifiers_to_use = []
        if args.classifier == 'svm':
            classifiers_to_use = ['svm']
        elif args.classifier == 'rf':
            classifiers_to_use = ['rf']
        elif args.classifier == 'xgb':
            classifiers_to_use = ['xgb']
        elif args.classifier == 'all':
            classifiers_to_use = ['svm', 'rf', 'xgb']

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
            
            # Apply data augmentation to training data
            if model is not None:
                train_meta_aug, train_embeddings_aug = apply_data_augmentation(
                    train_meta, train_embeddings, model, feature_extractor, device,
                    layer_names, args.model_type, args.augmentation_factor, args.minority_threshold
                )
            else:
                train_meta_aug = train_meta
                train_embeddings_aug = train_embeddings
            
            # Prepare data with potentially augmented embeddings
            X_train, y_train, _ = prepare_data(train_meta_aug, train_embeddings_aug, layer_name)
            X_test, y_test, _ = prepare_data(test_meta, test_embeddings, layer_name)
            
            if X_train is None or X_test is None:
                logger.warning(f"Skipping layer {layer_name} due to data preparation issues")
                continue

            # Create layer-specific results directory
            layer_results_dir = os.path.join(args.results_dir, f'layer_{layer_name}')
            os.makedirs(layer_results_dir, exist_ok=True)
            
            # Train models for each specified classifier
            for classifier_type in classifiers_to_use:
                logger.info(f"\n--- Training {classifier_type.upper()} for {layer_name} ---")
                
                # Train balanced model
                result = train_balanced_model(
                    X_train, y_train, X_test, y_test,
                    classifier_type=classifier_type,
                    smote_k_neighbors=args.smote_k_neighbors
                )
                
                if result is None:
                    logger.warning(f"Failed to train {classifier_type} model for {layer_name}")
                    continue
                
                # Store y_test for visualization purposes
                result['y_test'] = y_test
                
                # Create visualizations
                create_visualizations(result, y_test, layer_results_dir, layer_name, classifier_type)
                
                # Create classification report
                y_pred = result['Predictions']
                report = classification_report(y_test, y_pred)
                logger.info(f"\nClassification Report for {classifier_type.upper()} ({layer_name}):")
                logger.info(f"\n{report}")
                
                # Save classification report
                with open(os.path.join(layer_results_dir, f'classification_report_{classifier_type}.txt'), 'w') as f:
                    f.write(f"Balanced {classifier_type.upper()} model on {layer_name}\n")
                    f.write(f"Balanced Accuracy: {result['Balanced_Accuracy']:.4f}\n")
                    f.write(f"F1 Weighted: {result['F1_Weighted']:.4f}\n")
                    f.write(f"F1 Macro: {result['F1_Macro']:.4f}\n")
                    f.write(f"Data Augmentation: Yes\n")
                    f.write(f"SMOTE: Yes\n")
                    f.write(f"Class Weights: Yes\n\n")
                    f.write(report)
                
                # Save detailed results
                save_results(result, layer_results_dir, args.model_type, layer_name, classifier_type)
                
                # Save model
                model_path = save_model(result['Pipeline'], layer_name, args.model_type, 
                                      classifier_type, layer_results_dir)
                
                # Track best overall result
                layer_result = {
                    'Layer': layer_name,
                    'Classifier': classifier_type,
                    'Balanced_Accuracy': result['Balanced_Accuracy'],
                    'F1_Weighted': result['F1_Weighted'],
                    'F1_Macro': result['F1_Macro'],
                    'Model_Path': model_path
                }
                all_layer_results.append(layer_result)
                
                if result['Balanced_Accuracy'] > best_balanced_acc:
                    best_balanced_acc = result['Balanced_Accuracy']
                    best_model = result
                    best_layer = layer_name
                    best_classifier = classifier_type
        
        # Create overall comparison across layers and classifiers
        if all_layer_results:
            logger.info(f"\n{'='*80}")
            logger.info(f"CREATING OVERALL COMPARISON")
            logger.info(f"{'='*80}")
            
            results_df = pd.DataFrame(all_layer_results)
            
            # Save comparison
            results_df.to_csv(os.path.join(args.results_dir, 'all_results_comparison.csv'), index=False)
            
            # Create comparison visualization
            plt.figure(figsize=(15, 10))
            
            metrics = ['Balanced_Accuracy', 'F1_Weighted', 'F1_Macro']
            
            for i, metric in enumerate(metrics, 1):
                plt.subplot(2, 2, i)
                
                # Group by classifier if multiple classifiers used
                if len(classifiers_to_use) > 1:
                    for clf in classifiers_to_use:
                        clf_data = results_df[results_df['Classifier'] == clf]
                        if len(clf_data) > 0:
                            plt.plot(range(len(clf_data)), clf_data[metric], 
                                    marker='o', linewidth=2, markersize=8, label=clf.upper())
                            
                            # Set x-axis labels to layer names
                            plt.xticks(range(len(clf_data)), clf_data['Layer'], rotation=45)
                    plt.legend()
                else:
                    plt.plot(range(len(results_df)), results_df[metric], 
                            marker='o', linewidth=2, markersize=8)
                    plt.xticks(range(len(results_df)), results_df['Layer'], rotation=45)
                
                plt.title(f'{metric} by Layer', fontsize=14)
                plt.xlabel('Layer', fontsize=12)
                plt.ylabel(metric, fontsize=12)
                plt.grid(True, alpha=0.3)
                
                # Highlight best result
                if len(results_df) > 0:
                    best_idx = results_df[metric].idxmax()
                    best_value = results_df.iloc[best_idx][metric]
                    best_layer_name = results_df.iloc[best_idx]['Layer']
                    best_clf = results_df.iloc[best_idx]['Classifier']
                    
                    plt.annotate(f'Best: {best_layer_name}\n{best_clf.upper()}: {best_value:.3f}', 
                               xy=(best_idx, best_value), xytext=(10, 10),
                               textcoords='offset points', ha='left',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.results_dir, 'overall_comparison.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved overall comparison to {args.results_dir}")

    # Final summary
    if args.split == 'predefined' and best_model:
        logger.info(f"\n{'='*80}")
        logger.info(f"BEST OVERALL RESULTS")
        logger.info(f"{'='*80}")
        logger.info(f"Best Layer: {best_layer}")
        logger.info(f"Best Classifier: {best_classifier}")
        logger.info(f"Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}")
        logger.info(f"F1 (Weighted): {best_model['F1_Weighted']:.4f}")
        logger.info(f"F1 (Macro): {best_model['F1_Macro']:.4f}")
        
        # Save final summary
        with open(os.path.join(args.results_dir, 'final_summary.txt'), 'w') as f:
            f.write("=== FINAL EXPERIMENT SUMMARY ===\n\n")
            f.write(f"Dataset: {args.model_type} embeddings\n")
            f.write(f"Split strategy: {args.split}\n")
            f.write(f"Approach: Balanced (Data Augmentation + SMOTE + Class Weights)\n")
            f.write(f"Augmentation factor: {args.augmentation_factor}\n")
            f.write(f"Minority threshold: {args.minority_threshold}\n")
            f.write(f"SMOTE k-neighbors: {args.smote_k_neighbors}\n\n")
            f.write(f"Best overall configuration:\n")
            f.write(f"  Layer: {best_layer}\n")
            f.write(f"  Classifier: {best_classifier}\n")
            f.write(f"  Balanced Accuracy: {best_model['Balanced_Accuracy']:.4f}\n")
            f.write(f"  F1 Weighted: {best_model['F1_Weighted']:.4f}\n")
            f.write(f"  F1 Macro: {best_model['F1_Macro']:.4f}\n\n")
            f.write(f"Metrics Explanation:\n")
            f.write(f"- Balanced Accuracy: Average recall across all stuttering classes\n")
            f.write(f"- F1 Weighted: F1 score weighted by class frequency\n")
            f.write(f"- F1 Macro: Average F1 score across all classes\n")
            
        logger.info(f"Saved final summary to {args.results_dir}")

    logger.info("\n=== Balanced Model Training Complete ===")
    logger.info("All results, models, and visualizations have been saved.")

if __name__ == "__main__":
    main()