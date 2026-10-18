import os
from huggingface_hub import snapshot_download

# Download the specific 'lego' subdirectory from the dataset repository
local_dir = "./data/lego"
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="rishitdagli/nerf-gs-datasets",
    repo_type="dataset",
    allow_patterns=["lego/**"],  # Only download the lego directory files
    local_dir="./data_temp"
)

# Move files to your clean data path structure
import shutil
source_path = "./data_temp/lego"
if os.path.exists(source_path):
    for item in os.listdir(source_path):
        s = os.path.join(source_path, item)
        d = os.path.join(local_dir, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)
    shutil.rmtree("./data_temp")
    print("Lego dataset successfully downloaded and ready at ./data/lego!")
