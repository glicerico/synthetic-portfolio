"""Batch generate all dataset difficulties and upload them online for easy sharing."""

import os
import shutil
import zipfile
import subprocess
from pathlib import Path
from portfolio_eval.generate_dataset import main as gen_main

CONFIG_VARIANTS = {
    "easy": "configs/dataset_easy.yaml",
    "medium": "configs/dataset_medium.yaml",
    "hard": "configs/dataset_hard.yaml",
    "noise_trap": "configs/dataset_noise_trap.yaml",
    "turnover_trap": "configs/dataset_turnover_trap.yaml",
}

def zip_directory(src_dir: Path, zip_path: Path):
    """Zip the contents of src_dir into zip_path."""
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(src_dir):
            for file in files:
                file_path = Path(root) / file
                # Save relative to the directory itself
                arcname = file_path.relative_to(src_dir)
                zipf.write(file_path, arcname)

def upload_to_catbox(file_path: Path) -> str:
    """Upload file to catbox.moe using curl and return the link."""
    print(f"Uploading {file_path.name} to catbox.moe...")
    try:
        res = subprocess.run(
            [
                "curl", "-s", "-k",
                "-F", "reqtype=fileupload",
                "-F", f"fileToUpload=@{file_path}",
                "https://catbox.moe/user/api.php"
            ],
            capture_output=True,
            text=True,
            check=True
        )
        link = res.stdout.strip()
        if link.startswith("https://"):
            return link
        else:
            raise RuntimeError(f"Unexpected response from catbox.moe: {link}")
    except Exception as e:
        print(f"Error uploading {file_path.name}: {e}")
        return ""

def main(out_dir: str, share: bool):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    links = {}
    
    for name, config_file in CONFIG_VARIANTS.items():
        print(f"\n=========================================")
        print(f"Generating variant: {name.upper()}")
        print(f"=========================================")
        
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"Warning: Configuration file {config_file} not found. Skipping...")
            continue
            
        pub_out = out_path / f"public_{name}"
        hid_out = out_path / f"hidden_{name}"
        
        # Clean existing directories to avoid file accumulation
        if pub_out.exists():
            shutil.rmtree(pub_out)
        if hid_out.exists():
            shutil.rmtree(hid_out)
            
        # Generate dataset
        gen_main(str(config_path), str(pub_out), str(hid_out))
        
        # Zip public package
        zip_path = out_path / f"public_{name}.zip"
        if zip_path.exists():
            zip_path.unlink()
            
        print(f"Zipping {pub_out} to {zip_path}...")
        zip_directory(pub_out, zip_path)
        
        if share:
            link = upload_to_catbox(zip_path)
            if link:
                links[name] = link
                print(f"Successfully uploaded: {link}")
            else:
                print(f"Failed to upload {name} to catbox.moe")
                
    if share and links:
        print(f"\n=========================================")
        print(f"SHAREABLE DATASET LINKS")
        print(f"=========================================")
        for name, link in links.items():
            print(f"{name:<15} : {link}")
        print(f"=========================================")
