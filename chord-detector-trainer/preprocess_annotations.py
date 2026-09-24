import os
import glob
import re
import argparse
import traceback
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

# Fixed feature normalization constants (must match Kotlin)
DB_FLOOR = -80.0
DB_CEIL = 0.0


def parse_chord_string(chord_str):
    """
    Parses complex Isophonics chord strings (e.g. 'C:maj7', 'Bb/5', 'F:min', 'N')
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

    # Take the FIRST alternate interpretation before quality parsing.
    # Isophonics uses '|' and '(' for alternates, e.g. 'C:maj|C:min', 'C:(maj7|maj)'.
    # Without this, 'maj|C:min' would contain 'min' and be mislabeled.
    shorthand = re.split(r"[|(]", shorthand)[0].strip()

    # Determine major vs minor triad.
    # IMPORTANT: do NOT use `"m" in shorthand` because "m" is a substring of "maj".
    q = shorthand.lower()
    if q == "m" or q.startswith("min") or q.startswith("dim") or q.startswith("hdim"):
        quality = "min"
    else:
        # Default triads, maj, 7, aug, sus, etc. map to major
        quality = "maj"

    target_label = f"{root}:{quality}"
    return LABEL_TO_IDX.get(target_label, NO_CHORD_IDX)


# -------------------------------------------------------------------------
# 2. Audio Feature & Annotation Alignment
# -------------------------------------------------------------------------
def extract_cqt_features(audio_path, sr=22050, hop_length=512, n_bins=84):
    """
    Extracts CQT + Delta features matching the model input shape (3, 84, T).

    Uses FIXED normalization (dB clipped to [DB_FLOOR, DB_CEIL], scaled to [0,1])
    so the same transform is reproducible in Kotlin without storing per-song stats.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=hop_length,
        n_bins=n_bins, bins_per_octave=12
    ))

    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)

    # Fixed normalization: clip to [DB_FLOOR, 0] then scale to [0, 1]
    log_cqt = np.clip(log_cqt, DB_FLOOR, DB_CEIL)
    log_cqt = (log_cqt - DB_FLOOR) / (DB_CEIL - DB_FLOOR)

    delta1 = librosa.feature.delta(log_cqt)
    delta2 = librosa.feature.delta(log_cqt, order=2)

    features = np.stack([log_cqt, delta1, delta2], axis=0).astype(np.float32)
    return features


def parse_lab_file(lab_path):
    """
    Parses Isophonics/MARL .lab files returning list of (start_time, end_time, class_idx)
    """
    annotations = []
    with open(lab_path, "r", encoding="utf-8") as f:
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

    # Sort by start time so overlapping segments resolve deterministically
    for start_time, end_time, class_idx in sorted(annotations, key=lambda a: a[0]):
        start_frame = int(np.round((start_time * sr) / hop_length))
        end_frame = int(np.round((end_time * sr) / hop_length))

        start_frame = max(0, min(start_frame, total_frames))
        end_frame = max(0, min(end_frame, total_frames))

        frame_labels[start_frame:end_frame] = class_idx

    return frame_labels


# -------------------------------------------------------------------------
# 3. Batch Processing Dataset Pipeline
# -------------------------------------------------------------------------
def process_dataset(audio_dir, lab_dir, output_dir, overwrite=False):
    """
    Walks through audio directory, matches corresponding .lab files, extracts
    features and labels, and saves compressed numpy arrays (.npz).
    """
    os.makedirs(output_dir, exist_ok=True)

    valid_extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"]
    audio_files = []
    for ext in valid_extensions:
        audio_files.extend(glob.glob(os.path.join(audio_dir, f"**/{ext}"), recursive=True))

    print(f"Found {len(audio_files)} audio tracks to process...")

    processed_count = 0
    skipped_count = 0
    failed_count = 0

    for audio_path in audio_files:
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        out_file = os.path.join(output_dir, f"{filename}.npz")

        if os.path.exists(out_file) and not overwrite:
            skipped_count += 1
            continue

        # Search for matching .lab file
        lab_path = os.path.join(lab_dir, f"{filename}.lab")
        if not os.path.exists(lab_path):
            matching_labs = glob.glob(os.path.join(lab_dir, f"**/{filename}.lab"), recursive=True)
            if matching_labs:
                lab_path = matching_labs[0]
            else:
                print(f"Skipping {filename}: Missing matching .lab file.")
                failed_count += 1
                continue

        try:
            features = extract_cqt_features(audio_path)
            total_frames = features.shape[2]

            annotations = parse_lab_file(lab_path)
            labels = create_frame_labels(annotations, total_frames)

            np.savez_compressed(out_file, x=features, y=labels)

            processed_count += 1
            print(f"[{processed_count}] Processed: {filename} | Frames: {total_frames}")

        except Exception:
            failed_count += 1
            print(f"Error processing {filename}:")
            traceback.print_exc()

    print(f"\nDone. Processed={processed_count}, Skipped={skipped_count}, Failed={failed_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess audio and .lab files into training tensors.")
    parser.add_argument("--audio_dir", type=str, default="./dataset/audio", help="Directory containing audio files")
    parser.add_argument("--lab_dir", type=str, default="./dataset/annotations", help="Directory containing .lab files")
    parser.add_argument("--output_dir", type=str, default="./dataset/processed", help="Directory to save .npz files")
    parser.add_argument("--overwrite", action="store_true", help="Reprocess files that already exist")

    args = parser.parse_args()

    if os.path.exists(args.audio_dir) and os.path.exists(args.lab_dir):
        process_dataset(args.audio_dir, args.lab_dir, args.output_dir, overwrite=args.overwrite)
    else:
        print("Directory setup check:")
        print(f"1. Place audio files in: {args.audio_dir}")
        print(f"2. Place corresponding .lab files in: {args.lab_dir}")
        print("3. Run: python preprocess_annotations.py --audio_dir <path> --lab_dir <path> --output_dir <path>")