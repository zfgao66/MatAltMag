import os
import os.path as osp
import re
import pandas as pd

base_dir = 'save_files/'
alter_file = os.path.join(base_dir, "altermaganization.txt")
no_alter_file = os.path.join(base_dir, "no_altermaganization.txt")

# alter file
with open(alter_file, 'r') as file:
    file_content = file.read()
pattern = r"'(mp-\d+)'"

# Find all matches
matches = re.findall(pattern, file_content)

pd.DataFrame(matches).to_csv(osp.join(base_dir, 'candidate.csv'))

# no-alter file
with open(no_alter_file, 'r') as file:
    file_content = file.read()
pattern = r"'(mp-\d+)'"

# Find all matches
matches = re.findall(pattern, file_content)

pd.DataFrame(matches).to_csv(osp.join(base_dir, "label0.csv"))
