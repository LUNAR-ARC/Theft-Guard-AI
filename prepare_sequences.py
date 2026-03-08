import os
import cv2
import numpy as np
from tqdm import tqdm

FRAME_DIR = "data/frames"
SEQ_DIR = "data/sequences"
SEQ_LEN = 10 

def build_sequences():
    # 1. Ensure output directory exists
    os.makedirs(SEQ_DIR, exist_ok=True)
    
    found_any = False

    for category in ["shoplifting", "normal"]:
        cat_path = os.path.join(FRAME_DIR, category)
        
        # Check if the folder even exists
        if not os.path.exists(cat_path):
            print(f"Sir, I cannot find the folder: {cat_path}")
            continue
        
        images = sorted([f for f in os.listdir(cat_path) if f.endswith(('.jpg', '.png'))])
        
        if len(images) < SEQ_LEN:
            print(f"Sir, folder '{category}' only has {len(images)} images. Need at least {SEQ_LEN}.")
            continue
        
        found_any = True
        label = 1 if category == "shoplifting" else 0
        
        # Sir, the tqdm bar starts here
        for i in tqdm(range(0, len(images) - SEQ_LEN, 5), desc=f"Bundling {category} sequences"):
            sequence = []
            for j in range(SEQ_LEN):
                img_path = os.path.join(cat_path, images[i + j])
                img = cv2.imread(img_path)
                
                if img is None:
                    continue
                
                img = cv2.resize(img, (64, 64))
                img = img.transpose(2, 0, 1) / 255.0 # Convert to CHW format
                sequence.append(img)
            
            if len(sequence) == SEQ_LEN:
                seq_arr = np.array(sequence, dtype=np.float32)
                output_file = os.path.join(SEQ_DIR, f"{label}_{category}_{i}.npy")
                np.save(output_file, seq_arr)

    if not found_any:
        print("\n❌ CRITICAL ERROR: Sir, no sequences were created! Check if data/frames/ has images.")
    else:
        print("\n✅ SUCCESS: Sir, your sequences are ready in data/sequences/.")

if __name__ == "__main__":
    build_sequences()