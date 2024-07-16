import os

def list_files_in_folder(folder_path, output_file):
    # Open the output file in write mode
    with open(output_file, 'w') as file:
        # Walk through the folder
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                # Construct the relative file path
                relative_path = os.path.relpath(os.path.join(root, filename), folder_path)
                # Write the file path to the output file
                file.write(f"data/obj/{relative_path}\n")

# Specify the folder and output file
folder_path = 'C:/Users/Tyler/Documents/OpenCV Learning/build/darknet/x64/data/obj/images/train'
output_file = 'train.txt'

# Call the function
list_files_in_folder(folder_path, output_file)
