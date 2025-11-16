import os
import random

#Process your dataset obey to following format ( multiple labeling is possible)
label_dict = {"Example-Filename.txt": ["[EXAMPLE_LABEL]"]}

#input and output paths
raw_data_dir = "raw_data"
processed_data_dir = "processed_data"
output_file_path = os.path.join(processed_data_dir, "train.txt")


#if not exists create output folder
if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

#open the training file of model open with "write" mode
try:
    with open(output_file_path,  "w", encoding="utf-8") as outfile:
        print(f"Creating {output_file_path}...")

        processed_count = 0
        skipped_count = 0

        #with .items() split the dictionary (filename, label_list) pairs.
        label_list = list(label_dict.items())

        #shuffle the list in-place
        random.shuffle(label_list)
        print(f"Totally {len(label_list)} songs found and shuffled for training." + "\n")

        for filename, labels in label_list:

            #if label_dict is empty, skip this file
            if not labels:
                print(f"Skipping(label_dict is empty): {filename})")
                skipped_count += 1
                continue

            #format the labels
            #["[FLEX]", "[BATTLE]"] -> "[FLEX] [BATTLE] " (with whitespace end of the string) converting
            label_str = " ".join(labels) + " "

            #read the lyrics
            song_lyric_path = os.path.join(raw_data_dir, filename)
            try:
                with open(song_lyric_path, "r", encoding="utf-8") as infile:
                    song_lyric_content = infile.read()
            except FileNotFoundError:
                print(f"ERROR: {filename} cannot be found in {raw_data_dir} folder. Skipping.")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"ERROR(File Reading):{filename}: {e}):")
                skipped_count += 1
                continue

            #write output in training format
            #Write to the main train.txt in [LABELS] <Lyrics...> format.
            outfile.write(label_str + song_lyric_content)

            #2 line break after every song
            outfile.write("\n" + "\n" + "\n")

            processed_count += 1

    print("\n" + "="*30)
    print(f"Data Processing is done: {processed_count} / {skipped_count}")
    print(f"{output_file_path} is ready for model training.")

except IOError as e:
    print(f"CRITICAL ERROR: Main training file '{output_file_path}'cannot load: {e}.")









