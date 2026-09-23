import os
import glob
import re
import numpy as np
import librosa

# -------------------------------------------------------------------------
# 1. Label Schemas & Pitch Class Mapping
# -------------------------------------------------------------------------
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard 25-Class Schema: 12 Major + 12 Minor + No Chord
CHORD_LABELS = (
    [f"{p}:maj" for p in PITCH_CLASSES] + 
    [f"{p}:min" for p in PITCH_CLASSES] + 
    ["N"]
)

LABEL_TO_IDX = {label: i for i, label in enumerate(CHORD_LABELS)}
NO_CHORD_IDX = LABEL_TO_IDX["N"]

# Equivalency map for common enharmonics (flats to sharps)
ENHARMONIC_MAP = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B"
}

def parse_chord_string(chord_str):
    """
    Parses complex Isophonics chord strings (e.g., 'C:maj7', 'Bb/5', 'F:min', 'N')
    and simplifies them into one of 25 canonical classes.
    """
    chord_str = chord_str.strip()
    
    # Handle silence / no-chord / unknown
    if chord_str in ["N", "X", "none", "", "&pause"]:
        return NO_CHORD_IDX
    
    # Remove inversion notation (e.g. "C:maj/5" -> "C:maj")
    if "/" in chord_str:
        chord_str = chord_str.split("/")[0]
        
    # Split root and shorthand quality (e.g. "C:maj", "A:min", "F#")
    parts = chord_str.split(":")
    root = parts[0]
    shorthand = parts[1] if len(parts) > 1 else "maj"
    
    # Normalize flats to sharps
    if root in ENHARMONIC_MAP:
        root = ENHARMONIC_MAP[root]
        
    if root not in PITCH_CLASSES:
        return NO_CHORD_IDX

    # Determine major vs minor triad
    # Handles Isophonics shorthand variants: min, maj, dim, aug, 7, maj7, min7, etc.
    shorthand_clean = re.sub(r'\(.*?\)', '', shorthand) # Remove added intervals
    
    if any(m in shorthand_clean for m in ["min", "m", "dim", "min7", "m7"]):
        quality = "min"
    else:
        # Default triads, maj, 7, aug, etc. map to major
        quality = "maj"
        
    target_label = f"{root}:{quality}"
    return LABEL_TO_IDX.get(target_label, NO_CHORD_IDX)

# -------------------------------------------------------------------------
# 2. Audio Feature & Annotation Alignment
# -------------------------------------------------------------------------
def extract_cqt_features(audio_path, sr=22050, hop_length=512, n_bins=84):
    """
    Extracts CQT + Delta features matching the model input shape (3, 84, T).
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    cqt = np.abs(librosa.cqt(y, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=12))
    
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    delta1 = librosa.feature.delta(log_cqt)
    delta2 = librosa.feature.delta(log_cqt, order=2)
    
    features = np.stack([log_cqt, delta1, delta2], axis=0).astype(np.float32)
    
    # Per-channel standardization
    for c in range(features.shape[0]):
        mean = np.mean(features[c])
        std = np.std(features[c]) + 1e-6
        features[c] = (features[c] - mean) / std
        
    return features

def parse_lab_file(lab_path):
    """
    Parses Isophonics/MARL .lab files returning list of (start_time, end_time, class_idx)
    """
    annotations = []
    with open(lab_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            start_time = float(parts[0])
            end_time = float(parts[1])
            chord_str = parts[2]
            
            class_idx = parse_chord_string(chord_str)
            annotations.append((start_time, end_time, class_idx))
            
    return annotations

def create_frame_labels(annotations, total_frames, sr=22050, hop_length=512):
    """
    Maps segment annotations to frame-level integer target labels.
    """
    frame_labels = np.full(total_frames, NO_CHORD_IDX, dtype=np.int64)
    
    for start_time, end_time, class_idx in annotations:
        start_frame = int(np.round((start_time * sr) / hop_length))
        end_frame = int(np.round((end_time * sr) / hop_length))
        
        # Clamp boundaries
        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))
        
        frame_labels[start_frame:end_frame] = class_idx
        
    return frame_labels

# -------------------------------------------------------------------------
# 3. Batch Processing Dataset Pipeline
# -------------------------------------------------------------------------
def process_dataset(audio_dir, lab_dir, output_dir):
    """
    Walks through audio directory, matches corresponding .lab files, extracts
    features and labels, and saves compressed numpy arrays (.npz).
    """
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(audio_dir, "**/*.wav"), recursive=True) + \
                  glob.glob(os.path.join(audio_dir, "**/*.mp3"), recursive=True) + \
                  glob.glob(os.path.join(audio_dir, "**/*.flac"), recursive=True)

    print(f"Found {len(audio_files)} audio tracks to process...")
    
    processed_count = 0
    for audio_path in audio_files:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        
        # Search for matching .lab file
        lab_path = os.path.join(lab_dir, f"{filename}.lab")
        if not os.path.exists(lab_path):
            # Try recursive search if structures differ
            matching_labs = glob.glob(os.path.join(lab_dir, f"**/{filename}.lab"), recursive=True)
            if matching_labs:
                lab_path = matching_labs[0]
            else:
                print(f"Skipping {filename}: Missing matching .lab file.")
                continue

        try:
            # 1. Extract CQT features (3, 84, T)
            features = extract_cqt_features(audio_path)
            total_frames = features.shape[2]
            
            # 2. Parse .lab file and map to frame array (T,)
            annotations = parse_lab_file(lab_path)
            labels = create_frame_labels(annotations, total_frames)
            
            # 3. Save aligned pair to output path
            out_file = os.path.join(output_dir, f"{filename}.npz")
            np.savez_compressed(out_file, x=features, y=labels)
            
            processed_count += 1
            print(f"[{processed_count}/{len(audio_files)}] Processed: {filename} | Frames: {total_frames}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    # Local test paths or dataset paths
    AUDIO_DIRECTORY = "./dataset/audio"
    LAB_DIRECTORY = "./dataset/annotations"
    OUTPUT_DIRECTORY = "./dataset/processed"
    
    if os.path.exists(AUDIO_DIRECTORY) and os.path.exists(LAB_DIRECTORY):
        process_dataset(AUDIO_DIRECTORY, LAB_DIRECTORY, OUTPUT_DIRECTORY)
    else:
        print("Directory setup check:")
        print("1. Place your track audio (.wav/.mp3) in ./dataset/audio")
        print("2. Place corresponding Isophonics/MARL .lab files in ./dataset/annotations")
        print("3. Run this script to generate preprocessed .npz tensors for PyTorch training.")