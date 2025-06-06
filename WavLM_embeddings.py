#!/usr/bin/env python3
# wavlm_embeddings_extractor.py - Extract WavLM model embeddings for stuttering classification

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/wavlm_embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract WavLM embeddings for stuttering classification')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory with KST data (should contain train/, test/, devel/ subdirectories)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--model_name', type=str, default='microsoft/wavlm-large',
                        choices=['microsoft/wavlm-base', 'microsoft/wavlm-base-plus', 
                                 'microsoft/wavlm-large', 'microsoft/wavlm-large-v2'],
                        help='WavLM model name (base, base-plus, large, large-v2)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'test', 'devel', 'all'],
                        help='Which dataset split to process')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N files')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cuda:1, cpu)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--max_length', type=int, default=None,
                        help='Maximum audio length in seconds (longer files will be trimmed)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Target sample rate for audio files')
    return parser.parse_args()

def verify_model_loading(model_name, model, feature_extractor, device):
    """Verify that the model was loaded correctly with the right size"""
    logger.info(f"Verifying model: {model_name}")
    
    # Check model type
    logger.info(f"Model type: {type(model).__name__}")
    
    # Check model config
    logger.info(f"Model config: {model.config}")
    
    # Check hidden size (should be 768 for base, 1024 for large)
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else "Unknown"
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Verify with a simple forward pass
    dummy_audio = np.zeros(16000)  # 1 second of silence
    inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt").to(device)
    
    with torch.no_grad():
        # Get encoder output shape
        outputs = model(inputs.input_values, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state
        logger.info(f"Last hidden state shape: {last_hidden_state.shape}")
        
        # Check hidden state dimension (should match hidden_size)
        logger.info(f"Hidden state dimension: {last_hidden_state.shape[-1]}")
    
    # Make sure the model on the device is actually the right size
    logger.info(f"Model is on device: {next(model.parameters()).device}")
    
    return hidden_size

def load_audio(file_path, target_sr=16000, max_length=None):
    """
    Load audio file and convert to the required format for WavLM
    
    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate
        max_length: Maximum length in seconds (None for no limit)
        
    Returns:
        Audio array as numpy array, or None if loading failed
    """
    try:
        # Try loading with torchaudio
        waveform, sample_rate = torchaudio.load(file_path)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != target_sr:
            resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
            waveform = resampler(waveform)

        # Trim if max_length is specified
        if max_length is not None:
            max_samples = int(max_length * target_sr)
            if waveform.shape[1] > max_samples:
                logger.info(f"Trimming audio from {waveform.shape[1]/target_sr:.2f}s to {max_length:.2f}s")
                waveform = waveform[:, :max_samples]

        # Log shape of processed audio
        logger.debug(f"Audio shape: {waveform.shape}, duration: {waveform.shape[1]/target_sr:.2f}s")
        
        return waveform.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def create_metadata_from_files(data_dir, split='all'):
    """Create metadata DataFrame from available files for KSF dataset structure"""
    # Actual KSF structure: /home/data/KST/compare22-KSF-full/
    #                       |-- wav/
    #                       |-- lab/
    #                       |-- features/
    
    # Define paths
    wav_dir = os.path.join(data_dir, 'wav')
    lab_dir = os.path.join(data_dir, 'lab')
    
    if not os.path.exists(wav_dir):
        logger.error(f"WAV directory not found: {wav_dir}")
        return pd.DataFrame()
    
    logger.info(f"Processing WAV files from: {wav_dir}")
    
    # Load labels if available
    label_mapping = {}
    
    # Try to find CSV files in the lab directory that might contain labels
    if os.path.exists(lab_dir):
        logger.info(f"Looking for label files in: {lab_dir}")
        csv_files = [f for f in os.listdir(lab_dir) if f.endswith('.csv')]
        
        for csv_file in csv_files:
            try:
                # Load CSV file
                csv_path = os.path.join(lab_dir, csv_file)
                df = pd.read_csv(csv_path)
                logger.info(f"Loaded labels from {csv_file}: {len(df)} entries")
                
                # Determine split based on filename
                current_split = 'unknown'
                if 'train' in csv_file:
                    current_split = 'train'
                elif 'test' in csv_file:
                    current_split = 'test'
                elif 'devel' in csv_file:
                    current_split = 'devel'
                
                # Skip if not in requested split (unless all splits requested)
                if split != 'all' and current_split != 'unknown' and current_split != split:
                    logger.info(f"Skipping {csv_file} (not in requested split: {split})")
                    continue
                
                # Extract file ID and label
                # Assuming first column is ID and second is label - adjust if different
                id_col = df.columns[0]
                label_col = None
                
                # Try to find a "label" column
                for col in df.columns:
                    if 'label' in col.lower():
                        label_col = col
                        break
                
                # If no label column found, use second column as default if available
                if label_col is None and len(df.columns) > 1:
                    label_col = df.columns[1]
                
                if label_col:
                    for idx, row in df.iterrows():
                        file_id = str(row[id_col])
                        # Add .wav if not already present
                        if not file_id.endswith('.wav'):
                            file_id = f"{file_id}.wav"
                        
                        label_mapping[file_id] = {
                            'label': row[label_col],
                            'split': current_split
                        }
                    
                    logger.info(f"Added {len(df)} label mappings from {csv_file}")
            except Exception as e:
                logger.error(f"Error processing label file {csv_file}: {e}")
    else:
        logger.warning(f"Label directory not found: {lab_dir}")
    
    # Get all WAV files
    all_data = []
    
    for root, dirs, files in os.walk(wav_dir):
        for file in files:
            if file.endswith('.wav'):
                wav_path = os.path.join(root, file)
                
                # Create entry for this file
                entry = {
                    'filename': os.path.splitext(file)[0],
                    'path': wav_path,
                }
                
                # Add label and split if available
                if file in label_mapping:
                    entry.update(label_mapping[file])
                elif os.path.basename(file) in label_mapping:
                    entry.update(label_mapping[os.path.basename(file)])
                else:
                    # Try to determine split from filename if no label mapping
                    if 'train' in file:
                        entry['split'] = 'train'
                    elif 'test' in file:
                        entry['split'] = 'test'
                    elif 'devel' in file:
                        entry['split'] = 'devel'
                    else:
                        entry['split'] = 'unknown'
                
                # Skip if not in requested split (unless all splits requested)
                if split != 'all' and entry.get('split', 'unknown') != split:
                    continue
                
                all_data.append(entry)
    
    if not all_data:
        logger.warning(f"No audio files found in {wav_dir} for the requested split: {split}")
    else:
        logger.info(f"Found {len(all_data)} WAV files in {wav_dir}")
    
    return pd.DataFrame(all_data)

def get_model_layer_info(model, feature_extractor, device):
    """Show information about the WavLM model's layers"""
    # Create dummy input
    dummy_audio = np.zeros(16000)  # 1 second of silence
    inputs = feature_extractor(dummy_audio, sampling_rate=16000, return_tensors="pt").to(device)

    # Get hidden states
    with torch.no_grad():
        outputs = model(inputs.input_values, output_hidden_states=True)
        
    hidden_states = outputs.hidden_states

    logger.info(f"Total layers in WavLM model (including input embeddings): {len(hidden_states)}")
    for i, hidden_state in enumerate(hidden_states):
        logger.info(f"Layer {i}: shape {hidden_state.shape}")
    
    return len(hidden_states)

def extract_wavlm_embeddings(audio_file, model, feature_extractor, device, layer_indices, max_length=None, sample_rate=16000):
    """
    Extract embeddings from the WavLM model for a given audio file.
    
    Args:
        audio_file: Path to the audio file
        model: WavLM model
        feature_extractor: Feature extractor for the model
        device: Device to use
        layer_indices: List of layer indices to extract embeddings from
        max_length: Maximum audio length in seconds (None for no limit)
        sample_rate: Target sample rate
        
    Returns:
        Dictionary with embeddings from specified layers
    """
    # Load audio
    audio_array = load_audio(audio_file, target_sr=sample_rate, max_length=max_length)
    if audio_array is None:
        return None

    # Process audio to get input features
    inputs = feature_extractor(
        audio_array,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).to(device)

    # Check for very long inputs that might cause memory issues
    input_length = inputs.input_values.shape[1]
    if input_length > 500000:  # Roughly 30 seconds at 16kHz
        logger.warning(f"Very long input ({input_length} samples, ~{input_length/sample_rate:.2f}s). This may cause memory issues.")
    
    try:
        # Extract embeddings with output_hidden_states=True
        with torch.no_grad():
            outputs = model(
                inputs.input_values,
                output_hidden_states=True,
                return_dict=True
            )

        # Get hidden states (all layers)
        hidden_states = outputs.hidden_states

        # Extract selected layers
        embeddings = {}
        for idx in layer_indices:
            if idx < len(hidden_states):
                # Get hidden state for this layer
                hidden_state = hidden_states[idx]

                # Average across time dimension to get a fixed-size representation
                # Shape: [batch_size, sequence_length, hidden_size] -> [batch_size, hidden_size]
                embedding = torch.mean(hidden_state, dim=1).cpu().numpy()

                embeddings[f"layer_{idx}"] = embedding.flatten()
            else:
                logger.warning(f"Layer {idx} is out of range (max: {len(hidden_states)-1})")

        return embeddings
    
    except RuntimeError as e:
        # Handle CUDA out of memory errors specially
        if "CUDA out of memory" in str(e):
            logger.error(f"CUDA out of memory while processing {audio_file}. Try using --max_length parameter to limit audio length.")
            # Clean up GPU memory
            torch.cuda.empty_cache()
            return None
        else:
            logger.error(f"Runtime error processing {audio_file}: {e}")
            return None
    except Exception as e:
        logger.error(f"Error extracting embeddings for {audio_file}: {e}")
        return None

def save_embeddings(embeddings_df, output_dir, split=None, expected_dim=None):
    """Save the extracted embeddings to disk"""
    if len(embeddings_df) == 0:
        logger.warning("No embeddings to save")
        return

    # Create output directory for the split if needed
    if split and split != 'all':
        split_dir = os.path.join(output_dir, split)
        os.makedirs(split_dir, exist_ok=True)
    else:
        split_dir = output_dir
    
    # Save metadata (without the embedding vectors)
    metadata_cols = [col for col in embeddings_df.columns if not col.startswith('layer_')]
    metadata_df = embeddings_df[metadata_cols].copy()
    metadata_df.to_csv(os.path.join(split_dir, 'embedding_metadata.csv'), index=False)
    logger.info(f"Saved metadata for {len(metadata_df)} files to {split_dir}")

    # Save each layer's embeddings separately as numpy arrays
    embedding_cols = [col for col in embeddings_df.columns if col.startswith('layer_')]
    
    # Log all embedding columns that will be saved
    logger.info(f"Embedding columns to save: {embedding_cols}")
    
    for col in embedding_cols:
        try:
            # Verify dimensions if expected_dim is provided
            if expected_dim is not None and len(embeddings_df) > 0:
                sample_embedding = embeddings_df[col].iloc[0]
                actual_dim = len(sample_embedding)
                if actual_dim != expected_dim:
                    logger.warning(f"WARNING: {col} has dimension {actual_dim} but expected {expected_dim}")
            
            # Print a sample of the column to verify format
            if len(embeddings_df) > 0:
                sample_shape = np.array(embeddings_df[col].iloc[0]).shape
                logger.info(f"Sample shape for {col}: {sample_shape}")
            
            embeddings = np.stack(embeddings_df[col].values)
            np.save(os.path.join(split_dir, f"{col}_embeddings.npy"), embeddings)
            logger.info(f"Saved {col} embeddings with shape {embeddings.shape}")
        except Exception as e:
            logger.error(f"Error saving {col} embeddings: {e}")
            logger.error(f"Exception details: {str(e)}")

def save_checkpoint(results, output_dir, split, checkpoint_num):
    """Save a checkpoint of the current results"""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save as pickle for easy resuming
    import pickle
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{split}_{checkpoint_num}.pkl')
    
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Saved checkpoint {checkpoint_num} for {split} split with {len(results)} processed files")

def load_checkpoint(output_dir, split, checkpoint_num):
    """Load a checkpoint of previous results"""
    checkpoint_path = os.path.join(output_dir, 'checkpoints', f'checkpoint_{split}_{checkpoint_num}.pkl')
    
    if not os.path.exists(checkpoint_path):
        logger.info(f"No checkpoint found at {checkpoint_path}")
        return []
    
    import pickle
    with open(checkpoint_path, 'rb') as f:
        results = pickle.load(f)
    
    logger.info(f"Loaded checkpoint {checkpoint_num} for {split} split with {len(results)} processed files")
    return results

def find_latest_checkpoint(output_dir, split):
    """Find the latest checkpoint number for a specific split"""
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    
    if not os.path.exists(checkpoint_dir):
        return None
    
    # Look for checkpoints for the specific split
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                      if f.startswith(f'checkpoint_{split}_') and f.endswith('.pkl')]
    
    if not checkpoint_files:
        return None
    
    # Extract checkpoint numbers and find max
    checkpoint_nums = [int(f.split('_')[-1].split('.')[0]) for f in checkpoint_files]
    return max(checkpoint_nums) if checkpoint_nums else None

def main():
    """Main execution function"""
    # Parse arguments
    args = parse_args()
    
    # Set up device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Command line arguments: {args}")
    
    # Verify data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory {args.data_dir} does not exist!")
        logger.error("Please check the path and try again.")
        return
    
    # Check for expected directory structure
    wav_dir = os.path.join(args.data_dir, 'wav')
    lab_dir = os.path.join(args.data_dir, 'lab')
    
    if not os.path.exists(wav_dir):
        logger.error(f"WAV directory not found: {wav_dir}")
        logger.error("Please check that your data directory contains a 'wav' subdirectory.")
        return
    
    logger.info(f"Found WAV directory: {wav_dir}")
    logger.info(f"Label directory {'exists' if os.path.exists(lab_dir) else 'not found'}: {lab_dir}")
    
    # Create output directory structure
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Create split directories if needed
    if args.split == 'all':
        os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, 'devel'), exist_ok=True)
    else:
        os.makedirs(os.path.join(args.output_dir, args.split), exist_ok=True)
    
    # Load the WavLM model
    logger.info(f"Loading WavLM model: {args.model_name}")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model_name)
    model = WavLMModel.from_pretrained(args.model_name).to(device)
    
    # Verify model was loaded correctly and get expected dimension
    hidden_size = verify_model_loading(args.model_name, model, feature_extractor, device)
    
    # Expected dimensions based on model size
    expected_dim = None
    if 'large' in args.model_name:
        expected_dim = 1024  # Large models have 1024 dimensions
    elif 'base' in args.model_name:
        expected_dim = 768   # Base models have 768 dimensions
    
    if expected_dim and hidden_size != expected_dim:
        logger.error(f"ERROR: Expected hidden size {expected_dim} but got {hidden_size}")
        logger.error("This indicates the wrong model was loaded. Exiting.")
        return
    
    # Get model layer information
    num_layers = get_model_layer_info(model, feature_extractor, device)
    
    # Choose which layers to extract
    # For WavLM models, there are multiple transformer layers plus the input embeddings
    # Extract the last 3 layers and the middle layer for comprehensive analysis
    layer_indices = [num_layers-1, num_layers-2, num_layers-3, num_layers//2]
    
    logger.info(f"Selected layers for extraction: {layer_indices}")
    
    # Free some memory after layer info check
    torch.cuda.empty_cache()
    
    # Create metadata from available files (now updated for KSF structure)
    logger.info(f"Creating metadata from files in {args.data_dir}")
    metadata_df = create_metadata_from_files(args.data_dir, args.split)
    logger.info(f"Created metadata for {len(metadata_df)} files")
    
    if len(metadata_df) == 0:
        logger.error("No files found to process. Please check the data directory structure.")
        logger.error(f"Expected structure: {args.data_dir}/[train|test|devel]/")
        return
    
    # Display split distribution
    if 'split' in metadata_df.columns:
        logger.info("Split distribution:")
        for split_name, count in metadata_df['split'].value_counts().items():
            logger.info(f"  {split_name}: {count} files")
    
    # Process each split separately
    if args.split == 'all':
        splits_to_process = ['train', 'test', 'devel']
    else:
        splits_to_process = [args.split]
    
    for current_split in splits_to_process:
        logger.info(f"\nProcessing {current_split} split")
        
        # Filter metadata for current split
        if 'split' in metadata_df.columns:
            split_metadata = metadata_df[metadata_df['split'] == current_split]
        else:
            logger.warning(f"No split column found in metadata. Processing all {len(metadata_df)} files as {current_split}.")
            split_metadata = metadata_df
        
        logger.info(f"Found {len(split_metadata)} files for {current_split} split")
        
        if len(split_metadata) == 0:
            logger.warning(f"No files found for {current_split} split. Skipping.")
            continue
        
        # Check for existing progress if resuming
        results = []
        latest_checkpoint = None
        
        if args.resume:
            latest_checkpoint = find_latest_checkpoint(args.output_dir, current_split)
            if latest_checkpoint is not None:
                # Resume from checkpoint
                results = load_checkpoint(args.output_dir, current_split, latest_checkpoint)
                processed_files = set(item['path'] for item in results if 'path' in item)
                logger.info(f"Resuming from checkpoint {latest_checkpoint} with {len(processed_files)} already processed files")
                
                # Filter out already processed files
                split_metadata = split_metadata[~split_metadata['path'].isin(processed_files)]
            else:
                logger.info(f"No checkpoints found for {current_split} split. Starting fresh.")
        else:
            logger.info(f"Starting new extraction process for {current_split} split")
        
        # Process files for this split
        files_to_process = split_metadata.to_dict('records')
        logger.info(f"Will process {len(files_to_process)} files for {current_split} split")
        
        # Process in batches
        batch_size = args.batch_size
        checkpoint_counter = latest_checkpoint + 1 if latest_checkpoint is not None else 0
        
        for i in range(0, len(files_to_process), batch_size):
            batch = files_to_process[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(files_to_process)-1)//batch_size + 1}")
            
            batch_results = []
            for idx, row in enumerate(tqdm(batch)):
                try:
                    # Extract embeddings with max_length handling
                    embeddings = extract_wavlm_embeddings(
                        row['path'],
                        model,
                        feature_extractor,
                        device,
                        layer_indices,
                        max_length=args.max_length,
                        sample_rate=args.sample_rate
                    )
                    
                    if embeddings is None:
                        logger.warning(f"Failed to extract embeddings for {row['path']}")
                        continue
                    
                    # Store result
                    result = row.copy()  # Keep all original metadata
                    
                    # Add embeddings
                    for layer_name, embedding in embeddings.items():
                        result[layer_name] = embedding
                    
                    # Verify embedding dimension if expected
                    if expected_dim:
                        for layer_name, embedding in embeddings.items():
                            actual_dim = len(embedding)
                            if actual_dim != expected_dim:
                                logger.warning(f"WARNING: {layer_name} has dimension {actual_dim} but expected {expected_dim}")
                    
                    batch_results.append(result)
                    results.append(result)
                    logger.debug(f"Successfully extracted embeddings for {row['filename']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {row['path']}: {e}")
                    logger.error(f"Exception details: {str(e)}")
            
            # Verify dimensions of sample embeddings from the batch
            if batch_results:
                first_result = batch_results[0]
                for key, value in first_result.items():
                    if key.startswith('layer_'):
                        logger.info(f"Sample {key} dimension: {len(value)}")
            
            # Free up memory after each batch
            torch.cuda.empty_cache()
            
            # Save checkpoint after each batch
            if ((i + batch_size) % args.checkpoint_interval == 0) or ((i + batch_size) >= len(files_to_process)):
                save_checkpoint(results, args.output_dir, current_split, checkpoint_counter)
                checkpoint_counter += 1
        
        # Create DataFrame and save results for this split
        if results:
            results_df = pd.DataFrame(results)
            logger.info(f"Successfully extracted embeddings for {len(results_df)} files in {current_split} split")
            
            # Check embedding layer names before saving
            embedding_cols = [col for col in results_df.columns if col.startswith('layer_')]
            logger.info(f"Generated embedding columns: {embedding_cols}")
            
            # Save embeddings for this split
            save_embeddings(results_df, args.output_dir, current_split, expected_dim)
        else:
            logger.warning(f"No embeddings were extracted for {current_split} split")
    
    logger.info("\n=== WavLM Embedding Extraction Summary ===")
    logger.info(f"Model used: {args.model_name}")
    logger.info(f"Layers extracted: {layer_indices}")
    logger.info(f"Splits processed: {splits_to_process}")
    logger.info(f"Embeddings saved to: {args.output_dir}")

if __name__ == "__main__":
    main()