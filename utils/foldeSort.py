import os
import shutil

# Paths
wav_root = r"D:\Code research\DATASET V2\VCTK-Corpus\wav48"
txt_root = r"D:\Code research\DATASET V2\VCTK-Corpus\txt"
output_root = r"D:\Code research\Dataset-Sorted-Filtered"

# Ensure output directory exists
os.makedirs(output_root, exist_ok=True)

# Collect all transcriptions per phrase
phrase_transcripts = {}

for txt_file in os.listdir(txt_root):
    if txt_file.endswith(".txt"):
        parts = txt_file.split("_")
        if len(parts) < 2:
            continue
        phrase_num = parts[1].split(".")[0]
        phrase_key = f"phrase{phrase_num}"

        with open(os.path.join(txt_root, txt_file), "r", encoding="utf-8") as f:
            transcript = f.read().strip().lower()

        if phrase_key not in phrase_transcripts:
            phrase_transcripts[phrase_key] = set()
        phrase_transcripts[phrase_key].add(transcript)

# Keep only phrases where all transcriptions are identical
valid_phrases = {k for k, v in phrase_transcripts.items() if len(v) == 1}

print(f"Valid phrase folders with consistent text: {len(valid_phrases)}")

# Process WAV files
for speaker in os.listdir(wav_root):
    speaker_path = os.path.join(wav_root, speaker)
    if not os.path.isdir(speaker_path):
        continue

    for file in os.listdir(speaker_path):
        if not file.endswith(".wav"):
            continue

        parts = file.split("_")
        if len(parts) < 2:
            continue
        phrase_num = parts[1].split(".")[0]
        phrase_key = f"phrase{phrase_num}"

        if phrase_key in valid_phrases:
            phrase_folder = os.path.join(output_root, phrase_key)
            os.makedirs(phrase_folder, exist_ok=True)

            src_file = os.path.join(speaker_path, file)
            dest_file = os.path.join(phrase_folder, f"{speaker}_{file}")
            shutil.copy2(src_file, dest_file)

print("Filtered dataset organization complete.")
