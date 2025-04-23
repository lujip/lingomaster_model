import os

# Configurations
vctk_base_path = "D:/Code research/Dataset V2/VCTK-Corpus/txt"  # Path to VCTK 'txt' folder
output_wordlist = "D:/Code research/Renamed Audio/wordlist.txt"

# Phrase range (e.g., phrase001 to phrase050)
start_idx = 1
end_idx = 50

# Speaker list (sorted ensures consistent fallback order)
speakers = sorted([d for d in os.listdir(vctk_base_path) if os.path.isdir(os.path.join(vctk_base_path, d))])

# Create and write to wordlist
with open(output_wordlist, 'w', encoding='utf-8') as out_f:
    for phrase_num in range(start_idx, end_idx + 1):
        txt_suffix = f"{phrase_num:03d}"  # e.g., "015"
        found = False

        for speaker in speakers:
            txt_path = os.path.join(vctk_base_path, speaker, f"{speaker}_{txt_suffix}.txt")
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as txt_file:
                    sentence = txt_file.read().strip()
                    if sentence:
                        out_f.write(sentence + '\n')
                        print(f"✓ Phrase {txt_suffix} found in {speaker}")
                        found = True
                        break  # Move to the next phrase

        if not found:
            # If no speaker had this phrase, write placeholder
            out_f.write(f"[MISSING PHRASE {txt_suffix}]\n")
            print(f"⚠ Phrase {txt_suffix} not found in any speaker folder.")

print("\n✅ Finished generating wordlist.txt with fallback speakers.")
