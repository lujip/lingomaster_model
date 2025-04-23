import os

# Define the path to the txt folder
root_txt_folder = r"D:\Code research\DATASET V2\VCTK-Corpus\txt"  # Path to the txt folder

# Create a dictionary to store the phrase-to-text mapping
phrase_to_text = {}

# Iterate through all the txt files in the folder
for txt_file in os.listdir(root_txt_folder):
    if txt_file.endswith(".txt"):
        # Extract the phrase number from the filename (e.g., p225_027.txt => 027)
        phrase_num = txt_file.split("_")[1].split(".")[0]

        # Open the txt file to read its content
        with open(os.path.join(root_txt_folder, txt_file), "r") as f:
            phrase_text = f.read().strip()  # Read and clean up any whitespace

            # Store the phrase number and its corresponding text
            phrase_to_text[phrase_num] = phrase_text

# Print the phrase mappings
for phrase_num, phrase_text in phrase_to_text.items():
    print(f"Phrase {phrase_num}: \"{phrase_text}\"")
