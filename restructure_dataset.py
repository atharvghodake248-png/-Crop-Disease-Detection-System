import os, shutil

SRC = "dataset"
DST = "dataset_flat"

if os.path.exists(DST):
    shutil.rmtree(DST)
os.makedirs(DST)

for crop in os.listdir(SRC):
    crop_path = os.path.join(SRC, crop)
    if not os.path.isdir(crop_path):
        continue
    for disease in os.listdir(crop_path):
        disease_path = os.path.join(crop_path, disease)
        if not os.path.isdir(disease_path):
            continue
        new_class = f"{crop}_{disease}"
        shutil.copytree(disease_path, os.path.join(DST, new_class))
        print(f"Created class: {new_class}")

print("Done — flattened dataset is in dataset_flat/")