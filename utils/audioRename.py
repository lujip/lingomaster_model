import os
import shutil
import re

# Paths
base_path = "D:/Code research/To be Trained"         # phrase folders (e.g., phrase001)
wordlist_path = "D:/Code research/Renamed Audio/wordlist.txt"
output_path = "D:/Code research/Renamed Audio/p226 and up"  # Change folder name to reflect speakers

# Create output folder
os.makedirs(output_path, exist_ok=True)

# Load wordlist
with open(wordlist_path, 'r', encoding='utf-8') as f:
    wordlist = [line.strip() for line in f if line.strip()]

# Sanitize function
def sanitize_filename(text, max_length=100):
    text = text.replace(' ', '_')
    safe = re.sub(r'[\\/*?:"<>|]', '', text)
    return safe[:max_length]

# Valid speaker range (p226 to p360)
valid_speakers = [f'p{i}' for i in range(226, 361)]

# Processing
start_idx = 1
end_idx = 14

for idx in range(start_idx, end_idx + 1):
    folder_name = f'phrase{idx:03d}'
    folder_path = os.path.join(base_path, folder_name)

    if idx > len(wordlist):
        print(f"Index {idx} exceeds wordlist length.")
        break

    if not os.path.isdir(folder_path):
        print(f"⚠ Missing folder: {folder_name}")
        continue

    found = False
    for speaker in valid_speakers:
      #  target_filename = f"{speaker}_{(idx + 1):03d}.wav"
        target_filename = f"{speaker}_{idx:03d}.wav"
        source_path = os.path.join(folder_path, target_filename)
       # print (target_filename)
        if os.path.exists(source_path):
            phrase_text = wordlist[idx - 1]
            safe_name = sanitize_filename(phrase_text)
            dest_filename = f'{safe_name}.wav'
            dest_path = os.path.join(output_path, dest_filename)

            try:
                shutil.copy(source_path, dest_path)
                print(f"✓ Copied: {source_path} -> {dest_path}")
                found = True
                break
            except Exception as e:
                print(f"❌ Error copying {source_path}: {e}")
                found = True  # Still skip to next phrase if error occurs
                break

    if not found:
        print(f"⚠ Phrase {idx:03d}: No file found from p226 and up.")
