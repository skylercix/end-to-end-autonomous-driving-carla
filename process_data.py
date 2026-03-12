import os
import csv
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


INPUT_DIR = "dataset_manual"
OUTPUT_DIR = "dataset_processed"
SMOOTHING_WINDOW = 5  
KEEP_PROBABILITY = 0.1 
FLIP_PROBABILITY = 0.5 


FINAL_W, FINAL_H = 200, 66

def smooth_steering(steering_list):
    """
    Transformă input-ul de tastatură (0, -0.6) în ceva lin (0, -0.1, -0.3, -0.6)
    folosind o medie mobilă.
    """
    arr = np.array(steering_list)
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smoothed = np.convolve(arr, kernel, mode='same')
    return smoothed

def process_image_structure(img_pil):
    """
    Aplică decuparea cerului (Crop) și redimensionarea la 200x66 (Resize).
    Returnează imaginea micșorată ca RGB.
    """
    
    img_cropped = img_pil.crop((0, 80, 320, 240))
    
    img_resized = img_cropped.resize((FINAL_W, FINAL_H))
    return img_resized

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Șterg vechiul {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Procesez datele din '{INPUT_DIR}' -> '{OUTPUT_DIR}'...")
    print("Operații: Smooth + Filtrare + Crop + Resize + Flip")
    
    total_original = 0
    total_kept = 0
    
    all_original_angles = []
    all_new_angles = []

    for episode in os.listdir(INPUT_DIR):
        in_episode_path = os.path.join(INPUT_DIR, episode)
        if not os.path.isdir(in_episode_path):
            continue

        csv_file = os.path.join(in_episode_path, "controls_nav.csv")
        if not os.path.exists(csv_file):
            print(f"Sari peste {episode} (Nu exista controls_nav.csv)")
            continue
        
        rows = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for line in reader:
                rows.append(line)
        
        if not rows: continue

        steerings = [float(row[1]) for row in rows]
        all_original_angles.extend(steerings)
        
        smoothed_steerings = smooth_steering(steerings)

        out_episode_path = os.path.join(OUTPUT_DIR, episode)
        os.makedirs(out_episode_path)

        new_rows = []
        for i, row in enumerate(rows):
            img_name = row[0]
            new_steer = smoothed_steerings[i] 
            throttle = row[2]
            brake = row[3]
            command = int(row[4]) 

            
            if abs(new_steer) < 0.05:
                if random.random() > KEEP_PROBABILITY:
                    continue 

            src_img_path = os.path.join(in_episode_path, img_name)
            if not os.path.exists(src_img_path):
                continue

           
            img_pil = Image.open(src_img_path).convert("RGB")
            
            img_final = process_image_structure(img_pil)

            
            dst_img_path = os.path.join(out_episode_path, img_name)
            img_final.save(dst_img_path)
            new_rows.append([img_name, new_steer, throttle, brake, command])
            all_new_angles.append(new_steer)

            
            if random.random() < FLIP_PROBABILITY:
                flip_img_name = f"flip_{img_name}"
                dst_flip_path = os.path.join(out_episode_path, flip_img_name)
                
                img_flip = img_final.transpose(Image.FLIP_LEFT_RIGHT)
                img_flip.save(dst_flip_path)
           
                flip_steer = -new_steer
                flip_command = command
                if command == 1:
                    flip_command = 2
                elif command == 2:
                    flip_command = 1
                
                new_rows.append([flip_img_name, flip_steer, throttle, brake, flip_command])
                all_new_angles.append(flip_steer)

        
        with open(os.path.join(out_episode_path, "controls_nav.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(new_rows)
            
        total_original += len(rows)
        total_kept += len(new_rows)
        print(f"Episod {episode}: {len(rows)} -> {len(new_rows)} cadre.")

    print(f"\n--- REZULTAT FINAL ---")
    if total_original == 0:
        print("Nu au fost gasite date valide.")
        return
        
    print(f"Total cadre inițiale: {total_original}")
    print(f"Total cadre finale: {total_kept}")
    
    
    plt.figure(figsize=(14, 6)) 
    plt.subplot(1, 2, 1)
    plt.hist(all_original_angles, bins=25, color='#FF5733', edgecolor='black', alpha=0.7)
    plt.title("Original (Manual + Auto)")
    
    plt.subplot(1, 2, 2)
    plt.hist(all_new_angles, bins=55, color='#2ECC71', edgecolor='black', alpha=0.7)
    plt.title("Procesat (Smooth + Balanced + Flip)")
    
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    main()