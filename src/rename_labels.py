
import os
import shutil
from label_map import LABEL_MAP

def rename_dataset(input_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    for class_name in os.listdir(input_path):
        class_path = os.path.join(input_path, class_name)

        if not os.path.isdir(class_path):
            continue

        if class_name not in LABEL_MAP:
            print(f"Skipping: {class_name}")
            continue

        new_class = LABEL_MAP[class_name]
        new_class_path = os.path.join(output_path, new_class)

        os.makedirs(new_class_path, exist_ok=True)

        for img in os.listdir(class_path):
            src = os.path.join(class_path, img)
            dst = os.path.join(new_class_path, img)

            # Using shutil.copy2 to preserve metadata if desired
            shutil.copy2(src, dst)

    print(f"Done processing: {input_path}")
