#!/usr/bin/env python3
# whisper_embeddings_large.py - Extract Whisper large model embeddings for stuttering classification

import os
import torch
import torchaudio
import numpy as np
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from datetime import datetime
from transformers import WhisperProcessor, WhisperModel

# Setup logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/whisper_embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract Whisper embeddings for stuttering classification')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Base directory with KST data')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save embeddings')
    parser.add_argument('--model_name', type=str, default='openai/whisper-small',
                        help='Whisper model name (tiny, base, small, medium, large)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for processing')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'test', 'devel', 'all'],
                        help='Which dataset split to process')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='Save checkpoint every N files')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda:0, cuda:1, cpu)')
    return parser.parse_args()

def verify_model_loading(model_name, model, processor, device):
    """Verify that the model was loaded correctly with the right size"""
    logger.info(f"Verifying model: {model_name}")
    
    # Check model type
    logger.info(f"Model type: {type(model).__name__}")
    
    # Check model config
    logger.info(f"Model config: {model.config}")
    
    # Check hidden size (should be 768 for small, 1280 for large)
    hidden_size = model.config.hidden_size if hasattr(model.config, 'hidden_size') else "Unknown"
    logger.info(f"Model hidden size: {hidden_size}")
    
    # Verify with a simple forward pass
    dummy_audio = np.zeros(16000)  # 1 second of silence
    input_features = processor(dummy_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    
    with torch.no_grad():
        # Get encoder output shape
        encoder_output = model.encoder(input_features).last_hidden_state
        logger.info(f"Encoder output shape: {encoder_output.shape}")
        
        # Check encoder output dimension (should match hidden_size)
        logger.info(f"Encoder output dimension: {encoder_output.shape[-1]}")
    
    # Make sure the model on the device is actually the right size
    logger.info(f"Model is on device: {next(model.parameters()).device}")
    
    return hidden_size

def load_audio(file_path, target_sr=16000):
    """Load audio file and convert to the required format for Whisper"""
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

        return waveform.squeeze().numpy()
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

def create_metadata_from_files(data_dir, split='all'):
    """Create metadata DataFrame from available files"""
    wav_dir = os.path.join(data_dir, 'wav')
    lab_dir = os.path.join(data_dir, 'lab')
    
    # Find all available splits
    splits = []
    if split == 'all' or split == 'train':
        train_csv = os.path.join(lab_dir, 'train.csv')
        if os.path.exists(train_csv):
            splits.append(('train', train_csv))
    
    if split == 'all' or split == 'test':
        test_csv = os.path.join(lab_dir, 'test.csv')
        if os.path.exists(test_csv):
            splits.append(('test', test_csv))
    
    if split == 'all' or split == 'devel':
        devel_csv = os.path.join(lab_dir, 'devel.csv')
        if os.path.exists(devel_csv):
            splits.append(('devel', devel_csv))
    
    # Create combined DataFrame
    all_data = []
    
    for split_name, csv_path in splits:
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Add split column
            df['split'] = split_name
            
            # Process each row to add file path
            for idx, row in df.iterrows():
                # Look for filename in the dataframe
                # Assuming the ID is in the first column, get the filename
                if 'filename' in df.columns:
                    filename = row['filename']
                else:
                    # First column is likely the file ID
                    id_col = df.columns[0]
                    filename = f"{row[id_col]}.wav"
                
                # Check if file exists
                wav_path = os.path.join(wav_dir, filename)
                if not os.path.exists(wav_path):
                    # Try with split prefix (devel_0001.wav)
                    if not filename.startswith(f"{split_name}_"):
                        alt_filename = f"{split_name}_{filename}"
                        wav_path = os.path.join(wav_dir, alt_filename)
                        if os.path.exists(wav_path):
                            filename = alt_filename
                
                # Add to results if file exists
                full_path = os.path.join(wav_dir, filename)
                if os.path.exists(full_path):
                    row_dict = row.to_dict()
                    row_dict['path'] = full_path
                    row_dict['filename'] = os.path.splitext(filename)[0]
                    all_data.append(row_dict)
            
            logger.info(f"Loaded {len(df)} entries from {split_name} split")
            
        except Exception as e:
            logger.error(f"Error loading {csv_path}: {e}")
    
    # If no data loaded, try direct file listing
    if not all_data:
        logger.warning("No data loaded from CSV files. Trying direct file listing.")
        
        # Get all WAV files
        wav_files = []
        for root, dirs, files in os.walk(wav_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    
                    # Determine split
                    file_split = 'unknown'
                    if file.startswith('train_'):
                        file_split = 'train'
                    elif file.startswith('test_'):
                        file_split = 'test'
                    elif file.startswith('devel_'):
                        file_split = 'devel'
                    
                    # Skip if not in requested split
                    if split != 'all' and file_split != split:
                        continue
                    
                    wav_files.append({
                        'filename': os.path.splitext(file)[0],
                        'path': wav_path,
                        'split': file_split
                    })
        
        all_data = wav_files
        logger.info(f"Found {len(all_data)} WAV files by direct listing")
    
    return pd.DataFrame(all_data)

def get_model_layer_info(model, processor, device):
    """Show information about the Whisper model's encoder and decoder layers"""
    # Create dummy input
    dummy_audio = np.zeros(16000)  # 1 second of silence
    input_features = processor(dummy_audio, sampling_rate=16000, return_tensors="pt").input_features.to(device)

    # Get encoder hidden states
    with torch.no_grad():
        encoder_outputs = model.encoder(input_features, output_hidden_states=True, return_dict=True)

        # Prepare decoder inputs
        decoder_inputs = torch.zeros((1, 1), dtype=torch.long).to(device)  # Start token

        # Get decoder hidden states
        decoder_outputs = model.decoder(
            input_ids=decoder_inputs,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_hidden_states=True,
            return_dict=True
        )

    encoder_hidden_states = encoder_outputs.hidden_states
    decoder_hidden_states = decoder_outputs.hidden_states

    logger.info(f"Total encoder layers in Whisper model: {len(encoder_hidden_states)}")
    for i, hidden_state in enumerate(encoder_hidden_states):
        logger.info(f"Encoder Layer {i}: shape {hidden_state.shape}")

    logger.info(f"\nTotal decoder layers in Whisper model: {len(decoder_hidden_states)}")
    for i, hidden_state in enumerate(decoder_hidden_states):
        logger.info(f"Decoder Layer {i}: shape {hidden_state.shape}")
    
    return len(encoder_hidden_states), len(decoder_hidden_states)

def extract_whisper_embeddings_fixed(audio_file, model, processor, device, encoder_indices, decoder_indices):
    """Extract both encoder and decoder embeddings with explicit layer indices"""
    # Load audio
    audio_array = load_audio(audio_file)
    if audio_array is None:
        return None

    # Process audio to get input features
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

        # Prepare decoder inputs
        decoder_outputs = model.decoder(
            input_ids=torch.zeros((1, 1), dtype=torch.long).to(device),  # Start token
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            output_hidden_states=True,
            return_dict=True
        )

    # Get all hidden states
    encoder_states = encoder_outputs.hidden_states
    decoder_states = decoder_outputs.hidden_states

    # Initialize embeddings dictionary
    embeddings = {}

    # Extract encoder embeddings for specified layers
    for idx in encoder_indices:
        if idx < len(encoder_states):
            # Get the specific layer
            hidden_state = encoder_states[idx]
            
            # Average across time dimension
            embedding = torch.mean(hidden_state, dim=1).cpu().numpy()
            
            # Store with explicit layer name
            embeddings[f"encoder_layer_{idx}"] = embedding.flatten()
        else:
            logger.warning(f"Encoder layer {idx} is out of range (max: {len(encoder_states)-1})")

    # Extract decoder embeddings for specified layers
    for idx in decoder_indices:
        if idx < len(decoder_states):
            # Get the specific layer
            hidden_state = decoder_states[idx]
            
            # Get features from the first token
            embedding = hidden_state.squeeze(1).cpu().numpy()
            
            # Store with explicit layer name
            embeddings[f"decoder_layer_{idx}"] = embedding.flatten()
        else:
            logger.warning(f"Decoder layer {idx} is out of range (max: {len(decoder_states)-1})")

    return embeddings

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
    metadata_cols = [col for col in embeddings_df.columns
                     if not col.startswith('encoder_layer_') and not col.startswith('decoder_layer_')]
    metadata_df = embeddings_df[metadata_cols].copy()
    metadata_df.to_csv(os.path.join(split_dir, 'embedding_metadata.csv'), index=False)
    logger.info(f"Saved metadata for {len(metadata_df)} files to {split_dir}")

    # Save each layer's embeddings separately as numpy arrays
    embedding_cols = [col for col in embeddings_df.columns 
                     if col.startswith('encoder_layer_') or col.startswith('decoder_layer_')]
    
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
                    logger.warning("This suggests the embeddings were transformed somewhere in the pipeline")
            
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
    
    # Load the Whisper model with forced download
    logger.info(f"Loading Whisper model: {args.model_name}")
    try:
        # Try to force download the model to ensure we get the right one
        import shutil
        cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
        model_cache = os.path.join(cache_dir, f"models--openai--{args.model_name.split('/')[-1]}")
        if os.path.exists(model_cache):
            logger.info(f"Removing existing model cache at {model_cache}")
            shutil.rmtree(model_cache, ignore_errors=True)
    except Exception as e:
        logger.warning(f"Failed to clear model cache: {e}")
    
    # Load model with force_download=True
    processor = WhisperProcessor.from_pretrained(args.model_name, force_download=True)
    model = WhisperModel.from_pretrained(args.model_name, force_download=True).to(device)
    
    # Verify model was loaded correctly
    hidden_size = verify_model_loading(args.model_name, model, processor, device)
    expected_dim = 1280 if "large" in args.model_name.lower() else 768
    
    if "large" in args.model_name.lower() and hidden_size != 1280:
        logger.error(f"ERROR: Requested large model but got hidden size {hidden_size} instead of 1280")
        logger.error("This indicates the wrong model was loaded. Exiting.")
        return
    
    # Get model layer information
    num_encoder_layers, num_decoder_layers = get_model_layer_info(model, processor, device)
    
    # Choose which layers to extract (last three layers of encoder and decoder)
    # Use the actual layer indices based on model size
    encoder_layer_indices = [num_encoder_layers-1, num_encoder_layers-2, num_encoder_layers-3]
    decoder_layer_indices = [num_decoder_layers-1, num_decoder_layers-2, num_decoder_layers-3]
    
    logger.info(f"Selected encoder layers for extraction: {encoder_layer_indices}")
    logger.info(f"Selected decoder layers for extraction: {decoder_layer_indices}")
    
    # Free some memory after layer info check
    torch.cuda.empty_cache()
    
    # Create metadata from available files
    metadata_df = create_metadata_from_files(args.data_dir, args.split)
    logger.info(f"Created metadata for {len(metadata_df)} files")
    
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
        
        # Check for existing progress
        latest_checkpoint = find_latest_checkpoint(args.output_dir, current_split)
        results = []
        
        if latest_checkpoint is not None:
            # Resume from checkpoint
            results = load_checkpoint(args.output_dir, current_split, latest_checkpoint)
            processed_files = set(item['path'] for item in results if 'path' in item)
            logger.info(f"Resuming from checkpoint {latest_checkpoint} with {len(processed_files)} already processed files")
            
            # Filter out already processed files
            split_metadata = split_metadata[~split_metadata['path'].isin(processed_files)]
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
                    # Extract embeddings using the fixed function that explicitly uses layer indices
                    embeddings = extract_whisper_embeddings_fixed(
                        row['path'],
                        model,
                        processor,
                        device,
                        encoder_layer_indices,
                        decoder_layer_indices
                    )
                    
                    if embeddings is None:
                        logger.warning(f"Failed to extract embeddings for {row['path']}")
                        continue
                    
                    # Verify that the correct layer names are being used
                    layer_names = list(embeddings.keys())
                    logger.debug(f"Generated layer names: {layer_names}")
                    
                    # Store result
                    result = row.copy()  # Keep all original metadata
                    
                    # Add embeddings
                    for layer_name, embedding in embeddings.items():
                        result[layer_name] = embedding
                    
                    # Check embedding dimension
                    for layer_name, embedding in embeddings.items():
                        actual_dim = len(embedding)
                        if actual_dim != expected_dim:
                            logger.warning(f"WARNING: {layer_name} has dimension {actual_dim} but expected {expected_dim}")
                    
                    batch_results.append(result)
                    results.append(result)
                    logger.debug(f"Successfully extracted embeddings for {row['filename']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {row['path']}: {e}")
            
            # Verify dimensions of sample embeddings from the batch
            if batch_results:
                first_result = batch_results[0]
                for key, value in first_result.items():
                    if key.startswith('encoder_layer_') or key.startswith('decoder_layer_'):
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
            embedding_cols = [col for col in results_df.columns 
                            if col.startswith('encoder_layer_') or col.startswith('decoder_layer_')]
            logger.info(f"Generated embedding columns: {embedding_cols}")
            
            # Save embeddings for this split
            save_embeddings(results_df, args.output_dir, current_split, expected_dim)
        else:
            logger.warning(f"No embeddings were extracted for {current_split} split")
    
    logger.info("\n=== Whisper Embedding Extraction Summary ===")
    logger.info(f"Model used: {args.model_name}")
    logger.info(f"Encoder layers extracted: {encoder_layer_indices}")
    logger.info(f"Decoder layers extracted: {decoder_layer_indices}")
    logger.info(f"Splits processed: {splits_to_process}")
    logger.info(f"Embeddings saved to: {args.output_dir}")

if __name__ == "__main__":
    main()