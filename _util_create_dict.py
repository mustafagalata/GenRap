import os

raw_data_dir = "raw_data"

song_names = []

try:
    # with os.listdir() list the all files in the folder
    for filename in os.listdir(raw_data_dir):
        #just looking for .txt files
        if filename.endswith(".txt"):
            song_names.append(filename)

except FileNotFoundError:
    print(f"ERROR! '{raw_data_dir}' cannot find.")
    print(f"Please ensure that you execute the 'data_collection.py' file and collect the data.")
    exit()

#sort alphabetically
 #song_names.sort()

print("==========================================================================================")
print("Copy the following output and paste to the corresponding field in the  data_processing.py ")
print("==========================================================================================")

print("label_dict = {")
for filename in song_names:
    print(f"    {filename!r}: [],")
print("}\n")

print("#============================")
print("# Copying process is done .")
print("#============================")






