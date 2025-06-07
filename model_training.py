#!/usr/bin/env python3
# model_training.py - Train and evaluate stuttering classification models with class balancing and data augmentation

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
    parser = argparse.ArgumentParser(description='Train stuttering classification models with class balancing and data augmentation')
    
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
    parser.add_argument('--use_smote', action='store_true', default=True,
                        help='Use SMOTE oversampling')
    parser.add_argument('--use_class_weights', action='store_true', default=True,
                        help='Use class weights for imbalanced learning')
    parser.add_argument('--use_augmentation', action='store_true', default=True,
                        help='Use audio data augmentation for minority classes')
    parser.add_argument('--smote_k_neighbors', type=int, default=3,
                        help='Number of neighbors for SMOTE')
    parser.add_argument('--augmentation_factor', type=int, default=3,
                        help='How many augmented samples to create per minority sample')
    parser.add_argument('--minority_threshold', type=int, default=100,
                        help='Classes with fewer samples than this will be augmented')
    parser.add_argument('--model_name', type=str, default='microsoft/wavlm-large',
                        help='Model name for re-extracting embeddings from augmented audio (should match extraction script)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for model (cuda:0, cuda:1, cpu)')
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
    
    # Standard handling for non-predefined splits (same as before)
    else:
        logger.error("This updated script currently supports only predefined splits for augmentation")
        return None, {}

def augment_audio(waveform, sample_rate=16000, augmentation_type='random'):
    """
    Apply audio augmentation techniques suitable for stuttering data
    
    Args:
        waveform: Audio waveform as torch tensor
        sample_rate: Sample rate of the audio
        augmentation_type: Type of augmentation or 'random' for random selection
    
    Returns:
        Augmented waveform
    """
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    # Ensure proper shape [1, samples] for mono audio
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    
    # Random augmentation selection
    if augmentation_type == 'random':
        augmentation_type = random.choice(['speed', 'noise', 'pitch', 'volume'])
    
    try:
        if augmentation_type == 'speed':
            # Speed perturbation (0.9x to 1.1x speed)
            speed_factor = random.uniform(0.9, 1.1)
            # Resample to simulate speed change
            new_sample_rate = int(sample_rate * speed_factor)
            resampler = torchaudio.transforms.Resample(sample_rate, new_sample_rate)
            waveform_resampled = resampler(waveform)
            # Resample back to original rate
            resampler_back = torchaudio.transforms.Resample(new_sample_rate, sample_rate)
            waveform = resampler_back(waveform_resampled)
            
        elif augmentation_type == 'noise':
            # Add Gaussian noise (low level to preserve stuttering characteristics)
            noise_factor = random.uniform(0.005, 0.02)  # Very light noise
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
        
        return waveform.squeeze().numpy()
        
    except Exception as e:
        logger.warning(f"Augmentation failed: {e}. Returning original audio.")
        return waveform.squeeze().numpy()

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
        
        return waveform.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error loading audio {file_path}: {e}")
        return None

def extract_embeddings_from_audio(audio_array, model, feature_extractor, device, layer_indices):
    """Extract embeddings from augmented audio"""
    try:
        # Process audio
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        # Extract embeddings
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
                # Average across time dimension
                embedding = torch.mean(hidden_state, dim=1).cpu().numpy()
                embeddings[f"layer_{idx}"] = embedding.flatten()
        
        return embeddings
    except Exception as e:
        logger.error(f"Error extracting embeddings: {e}")
        return None

def extract_embeddings_from_audio_wavlm(audio_array, model, feature_extractor, device, layer_indices):
    """Extract WavLM embeddings from augmented audio"""
    try:
        # Process audio
        inputs = feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).to(device)
        
        # Extract embeddings
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
                # Average across time dimension
                embedding = torch.mean(hidden_state, dim=1).cpu().numpy()
                embeddings[f"layer_{idx}"] = embedding.flatten()
        
        return embeddings
    except Exception as e:
        logger.error(f"Error extracting WavLM embeddings: {e}")
        return None

def extract_embeddings_from_audio_whisper(audio_array, model, processor, device, layer_names):
    """Extract Whisper embeddings from augmented audio"""
    try:
        # Process audio for Whisper
        input_features = processor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_features.to(device)
        
        # Extract encoder embeddings
        with torch.no_grad():
            encoder_outputs = model.encoder(
                input_features,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Prepare decoder inputs for decoder embeddings
            decoder_outputs = model.decoder(
                input_ids=torch.zeros((1, 1), dtype=torch.long).to(device),
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                output_hidden_states=True,
                return_dict=True
            )
        
        encoder_states = encoder_outputs.hidden_states
        decoder_states = decoder_outputs.hidden_states
        embeddings = {}
        
        # Extract embeddings based on layer names
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
    """
    Apply data augmentation to minority classes
    
    Args:
        train_meta: Training metadata DataFrame
        train_embeddings: Dictionary of training embeddings
        model: Pre-trained model for embedding extraction
        feature_extractor: Feature extractor/processor for the model
        device: Device to use
        layer_names: List of layer names to extract embeddings from
        model_type: Type of model ('wavlm' or 'whisper')
        augmentation_factor: Number of augmented samples per original sample
        minority_threshold: Classes with fewer samples will be augmented
    
    Returns:
        Augmented metadata and embeddings
    """
    logger.info("\n=== Applying Data Augmentation ===")
    
    # Check if we have path information for audio files
    if 'path' not in train_meta.columns:
        logger.warning("No audio file paths found. Skipping data augmentation.")
        return train_meta, train_embeddings
    
    # Analyze class distribution
    if 'label' not in train_meta.columns:
        logger.warning("No labels found. Skipping data augmentation.")
        return train_meta, train_embeddings
    
    class_counts = train_meta['label'].value_counts()
    minority_classes = class_counts[class_counts < minority_threshold].index.tolist()
    
    logger.info(f"Classes to augment (< {minority_threshold} samples): {minority_classes}")
    logger.info(f"Augmentation factor: {augmentation_factor}")
    logger.info(f"Model type for augmentation: {model_type}")
    
    if not minority_classes:
        logger.info("No minority classes found. Skipping augmentation.")
        return train_meta, train_embeddings
    
    # Prepare lists for augmented data
    augmented_metadata = []
    augmented_embeddings = {layer: [] for layer in train_embeddings.keys()}
    
    # Process each minority class
    for class_name in minority_classes:
        class_samples = train_meta[train_meta['label'] == class_name]
        logger.info(f"Augmenting {len(class_samples)} samples for class '{class_name}'")
        
        for idx, row in class_samples.iterrows():
            # Load original audio
            audio_path = row['path']
            original_audio = load_audio_for_augmentation(audio_path)
            
            if original_audio is None:
                continue
            
            # Create multiple augmented versions
            for aug_idx in range(augmentation_factor):
                try:
                    # Apply random augmentation
                    augmented_audio = augment_audio(original_audio, augmentation_type='random')
                    
                    # Extract embeddings from augmented audio based on model type
                    if model_type.lower() == 'wavlm':
                        # For WavLM, convert layer names to indices
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
                    
                    # Create metadata entry for augmented sample
                    aug_metadata = row.copy()
                    aug_metadata['filename'] = f"{row['filename']}_aug_{aug_idx}"
                    aug_metadata['augmented'] = True
                    aug_metadata['augmentation_type'] = 'mixed'
                    
                    augmented_metadata.append(aug_metadata)
                    
                    # Store embeddings - only for layers that exist in original embeddings
                    for layer_name, embedding in aug_embeddings.items():
                        if layer_name in augmented_embeddings:
                            augmented_embeddings[layer_name].append(embedding)
                    
                except Exception as e:
                    logger.warning(f"Failed to augment sample {row['filename']}: {e}")
                    continue
    
    # Combine original and augmented data
    if augmented_metadata:
        # Convert augmented data to arrays
        for layer_name in augmented_embeddings:
            if augmented_embeddings[layer_name]:
                augmented_embeddings[layer_name] = np.array(augmented_embeddings[layer_name])
                logger.info(f"Created {len(augmented_embeddings[layer_name])} augmented embeddings for {layer_name}")
        
        # Combine metadata
        aug_meta_df = pd.DataFrame(augmented_metadata)
        combined_meta = pd.concat([train_meta, aug_meta_df], ignore_index=True)
        
        # Combine embeddings
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
    
        # Check for augmented samples
        if 'augmented' in metadata_df.columns:
            aug_counts = metadata_df.groupby(['label', 'augmented']).size().unstack(fill_value=0)
            logger.info("Original vs Augmented samples:")
            for label in aug_counts.index:
                original = aug_counts.loc[label, False] if False in aug_counts.columns else 0
                augmented = aug_counts.loc[label, True] if True in aug_counts.columns else 0
                logger.info(f"  {label}: {original} original + {augmented} augmented = {original + augmented} total")
    
        # Visualize label distribution
        plt.figure(figsize=(12, 8))
        
        if 'augmented' in metadata_df.columns:
            # Stacked bar chart showing original vs augmented
            aug_pivot = metadata_df.groupby(['label', 'augmented']).size().unstack(fill_value=0)
            aug_pivot.plot(kind='bar', stacked=True, 
                          color=['skyblue', 'orange'], 
                          title='Distribution of Stuttering Labels (Original vs Augmented)')
            plt.legend(['Original', 'Augmented'])
        else:
            label_counts.plot(kind='bar', title='Distribution of Stuttering Labels')
        
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

    # Check data quality
    check_data_quality(metadata_df, args.results_dir)

    # === PREDEFINED SPLITS WITH DATA AUGMENTATION ===
    if args.split == 'predefined':
        logger.info("Using predefined splits with optional data augmentation")

        # Separate metadata
        train_meta = metadata_df[metadata_df['split'] == 'train'].reset_index(drop=True)
        test_meta = metadata_df[metadata_df['split'].isin(['test', 'devel'])].reset_index(drop=True)
        
        logger.info(f"Train split has {len(train_meta)} samples")
        logger.info(f"Test split has {len(test_meta)} samples")

        # Load model for augmentation if needed
        model = None
        feature_extractor = None
        layer_names = []
        
        if args.use_augmentation:
            logger.info(f"Loading model for data augmentation: {args.model_name}")
            try:
                # Load model based on model type
                if args.model_type.lower() == 'wavlm':
                    from transformers import Wav2Vec2FeatureExtractor, WavLMModel
                    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
                    model = WavLMModel.from_pretrained(args.model_name).to(device)
                    
                    # Get layer names from available embeddings (WavLM format: layer_X)
                    layer_names = [layer for layer in embeddings.keys() if layer.startswith('layer_')]
                    logger.info(f"WavLM layer names for augmentation: {layer_names}")
                    
                elif args.model_type.lower() in ['whisper', 'whisper_large_fixed']:
                    from transformers import WhisperProcessor, WhisperModel
                    feature_extractor = WhisperProcessor.from_pretrained(args.model_name)
                    model = WhisperModel.from_pretrained(args.model_name).to(device)
                    
                    # Get layer names from available embeddings (Whisper format: encoder_layer_X, decoder_layer_X)
                    layer_names = [layer for layer in embeddings.keys() 
                                 if layer.startswith('encoder_layer_') or layer.startswith('decoder_layer_')]
                    logger.info(f"Whisper layer names for augmentation: {layer_names}")
                    
                else:
                    logger.warning(f"Data augmentation not supported for model type: {args.model_type}")
                    args.use_augmentation = False
                    
            except Exception as e:
                logger.error(f"Failed to load model for augmentation: {e}")
                logger.warning("Continuing without data augmentation.")
                args.use_augmentation = False

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
            
            # Apply data augmentation to training data
            if args.use_augmentation and model is not None:
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
                f.write(f"F1 Macro: {best_result['F1_Macro']:.4f}\n")
                f.write(f"Data Augmentation: {args.use_augmentation}\n")
                f.write(f"SMOTE: {args.use_smote}\n")
                f.write(f"Class Weights: {args.use_class_weights}\n\n")
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
        if args.use_augmentation:
            model_config += "_augmented"
        
        model_path = save_best_model(best_model['Pipeline'], best_layer, args.model_type, 
                                   model_config, os.path.join(args.results_dir, 'models'))
        
        # Save final summary
        with open(os.path.join(args.results_dir, 'final_summary.txt'), 'w') as f:
            f.write("=== FINAL EXPERIMENT SUMMARY ===\n\n")
            f.write(f"Dataset: {args.model_type} embeddings\n")
            f.write(f"Split strategy: {args.split}\n")
            f.write(f"Data Augmentation: {args.use_augmentation}\n")
            if args.use_augmentation:
                f.write(f"  Augmentation factor: {args.augmentation_factor}\n")
                f.write(f"  Minority threshold: {args.minority_threshold}\n")
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

    logger.info("\n=== Model Training and Evaluation Complete ===")
    logger.info("All results, models, and visualizations have been saved.")

if __name__ == "__main__":
    main()