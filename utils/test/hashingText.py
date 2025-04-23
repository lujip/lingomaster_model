import os
from collections import defaultdict

# Root folder containing subfolders like p225, p226, etc.
root_txt_folder = r"D:\Code research\DATASET V2\VCTK-Corpus\txt"

# phrase_number -> list of speakers
phrase_speakers = defaultdict(list)

# Loop through each speaker folder
for speaker_folder in os.listdir(root_txt_folder):
    speaker_path = os.path.join(root_txt_folder, speaker_folder)
    if not os.path.isdir(speaker_path):
        continue  # skip if not a folder

    # Check all .txt files inside the speaker's folder
    for file in os.listdir(speaker_path):
        if file.endswith(".txt"):
            phrase_number = file.split("_")[-1].split(".")[0]  # get "027" from "p225_027.txt"
            phrase_speakers[phrase_number].append(speaker_folder)

# Print the result
for phrase, speakers in sorted(phrase_speakers.items()):
    print(f"Phrase {phrase} is spoken by: {', '.join(speakers)}")

# Optional: Save to file
output_path = os.path.join(root_txt_folder, "phrase_speaker_map.txt")
with open(output_path, "w") as f:
    for phrase, speakers in sorted(phrase_speakers.items()):
        f.write(f"Phrase {phrase} is spoken by: {', '.join(speakers)}\n")

print(f"Mapping saved to '{output_path}'")
