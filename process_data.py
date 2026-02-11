import os
import csv
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURARE ---
INPUT_DIR = "dataset_manual"
OUTPUT_DIR = "dataset_processed"
SMOOTHING_WINDOW = 5  
KEEP_PROBABILITY = 0.3 # Pastram doar 30% din datele de mers drept

def smooth_steering(steering_list):
    """
    Transforma input-ul de tastatura (0, -0.6) in ceva lin (0, -0.1, -0.3, -0.6)
    folosind o medie mobila.
    """
    
    arr = np.array(steering_list)
    
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smoothed = np.convolve(arr, kernel, mode='same')
    return smoothed

def main():
    if os.path.exists(OUTPUT_DIR):
        print(f"Sterg vechiul {OUTPUT_DIR}...")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    print(f"Procesez datele din '{INPUT_DIR}' -> '{OUTPUT_DIR}'...")
    
    total_original = 0
    total_kept = 0
    
    
    all_original_angles = []
    all_new_angles = []

    for episode in os.listdir(INPUT_DIR):
        in_episode_path = os.path.join(INPUT_DIR, episode)
        if not os.path.isdir(in_episode_path):
            continue

        csv_file = os.path.join(in_episode_path, "controls.csv")
        if not os.path.exists(csv_file):
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
            old_steer = float(row[1])
            new_steer = smoothed_steerings[i] # Valoarea noua, lina
            throttle = row[2]
            brake = row[3]

            
            if abs(new_steer) < 0.05:
                if random.random() > KEEP_PROBABILITY:
                    continue 
            
            
            src_img = os.path.join(in_episode_path, img_name)
            dst_img = os.path.join(out_episode_path, img_name)
            
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
                new_rows.append([img_name, new_steer, throttle, brake])
                all_new_angles.append(new_steer)

        
        with open(os.path.join(out_episode_path, "controls.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(new_rows)
            
        total_original += len(rows)
        total_kept += len(new_rows)
        print(f"Episod {episode}: {len(rows)} -> {len(new_rows)} cadre.")

    print(f"\n--- REZULTAT FINAL ---")
    print(f"Total cadre initiale: {total_original}")
    print(f"Total cadre finale: {total_kept} (Reducere: {100-(total_kept/total_original*100):.1f}%)")
    
    
    plt.figure(figsize=(14, 6)) 
    
    # Grafic 1: ORIGINAL
    plt.subplot(1, 2, 1)
    plt.hist(all_original_angles, bins=25, color='#FF5733', edgecolor='black', alpha=0.7)
    plt.title(f"Original (Tastatură)\nTotal: {total_original} img", fontsize=14, fontweight='bold')
    plt.xlabel("Unghi Volan", fontsize=12)
    plt.ylabel("Numar de Imagini", fontsize=12) 
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Grafic 2: PROCESAT
    plt.subplot(1, 2, 2)
    plt.hist(all_new_angles, bins=55, color='#2ECC71', edgecolor='black', alpha=0.7)
    plt.title(f"Procesat (Smooth + Balansat)\nTotal: {total_kept} img", fontsize=14, fontweight='bold')
    plt.xlabel("Unghi Volan", fontsize=12)
    plt.ylabel("Numar de Imagini", fontsize=12) 
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout() 
    plt.show()

if __name__ == "__main__":
    main()