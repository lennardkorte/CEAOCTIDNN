
import os
import matplotlib.pyplot as plt

def count_files_in_subfolders(directory):
    file_counts = {}
    for root, dirs, files in os.walk(directory):
        # Consider the immediate subdirectories only
        if root == directory:
            continue
        file_count = len([file for file in files if os.path.isfile(os.path.join(root, file))])
        if file_count > 0:
            folder_name = os.path.basename(root)
            file_counts[folder_name] = file_count
    return file_counts

# Example usage
directory = "./data/h5s/original/ivoct_both_c2/orig"  # Example directory, replace with the actual directory path
file_counts = count_files_in_subfolders(directory)
print(file_counts)

for setname in ['set1', 'set2', 'set3', 'set32','set39', 'set63', 'set62', 'set28']:
    file_counts.pop(setname)

# Plotting a pie chart
plt.figure(figsize=(8, 6))
plt.pie(file_counts.values(), labels=file_counts.keys(), autopct='%1.1f%%', startangle=140)
plt.title('Number of Files in Each Folder')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

plot_filename = './pie_chart_files_distribution.png'
plt.savefig(plot_filename)