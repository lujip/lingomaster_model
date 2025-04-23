import os

# === CONFIG ===
sorted_dir = r"D:\Code research\Dataset-Sorted-2"
output_path = os.path.join(sorted_dir, "phrase_counts.txt")

# === COLLECT COUNTS ===
phrase_counts = []

for folder_name in os.listdir(sorted_dir):
    phrase_path = os.path.join(sorted_dir, folder_name)
    if os.path.isdir(phrase_path) and folder_name.startswith("phrase"):
        wav_count = sum(1 for file in os.listdir(phrase_path) if file.endswith(".wav"))
        phrase_counts.append((folder_name, wav_count))

# === SORT AND WRITE TO FILE ===
phrase_counts.sort(key=lambda x: x[1], reverse=True)  # Sort by count descending

with open(output_path, "w", encoding="utf-8") as f:
    for folder_name, count in phrase_counts:
        f.write(f"{folder_name}: {count} wav files\n")

print("✅ Done! Sorted phrase counts saved to 'phrase_counts.txt'")
