import os
import shutil
from collections import defaultdict

# === CONFIG ===
txt_dir = r"D:\Code research\DATASET V2\VCTK-Corpus\txt"
wav_dir = r"D:\Code research\DATASET V2\VCTK-Corpus\wav48"
output_dir = r"D:\Code research\Dataset-Sorted-2"
mapping_file_path = os.path.join(output_dir, "phrase_mapping.txt")

# === PREP ===
os.makedirs(output_dir, exist_ok=True)
phrase_map = defaultdict(list)
wav_lookup = {}

# STEP 1: Index all wav files with their full paths
for root, _, files in os.walk(wav_dir):
    for file in files:
        if file.endswith(".wav"):
            wav_lookup[file] = os.path.join(root, file)

# STEP 2: Recursively read all .txt files
for root, _, files in os.walk(txt_dir):
    for txt_filename in files:
        if txt_filename.endswith(".txt"):
            base = txt_filename.replace(".txt", "")
            wav_filename = base + ".wav"
            txt_path = os.path.join(root, txt_filename)

            with open(txt_path, "r", encoding="utf-8") as f:
                sentence = f.read().strip()

            phrase_map[sentence].append(wav_filename)

# STEP 3: Create folders, copy wavs, and save mapping
with open(mapping_file_path, "w", encoding="utf-8") as map_file:
    for idx, (sentence, wav_files) in enumerate(phrase_map.items(), start=1):
        phrase_folder = os.path.join(output_dir, f"phrase{idx:03d}")
        os.makedirs(phrase_folder, exist_ok=True)

        # Write mapping
        map_file.write(f"phrase{idx:03d}: {sentence}\n")

        for wav_filename in wav_files:
            if wav_filename in wav_lookup:
                shutil.copy(wav_lookup[wav_filename], os.path.join(phrase_folder, wav_filename))
            else:
                print(f"[!] Missing WAV: {wav_filename}")

print("✅ Done! Check `sorted_phrases/` and `phrase_mapping.txt`.")
