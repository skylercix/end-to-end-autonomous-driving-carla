import os
import csv
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


INPUT_DIR = "dataset_traffic"             
OUTPUT_DIR = "dataset_traffic_processed"  
SMOOTHING_WINDOW = 5  
KEEP_PROBABILITY = 0.1 
FLIP_PROBABILITY = 0.5 

FINAL_W, FINAL_H = 200, 66

def smooth_steering(steering_list):
    """
    Transformă input-ul de tastatură într-o curbă mai lină folosind o medie mobilă.
    """
    arr = np.array(steering_list)
    kernel = np.ones(SMOOTHING_WINDOW) / SMOOTHING_WINDOW
    smoothed = np.convolve(arr, kernel, mode='same')
    return smoothed

def process_image_structure(img_pil):
    """
    Aplică decuparea cerului (Crop) și redimensionarea la 200x66 (Resize).
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
    
    
    all_original_steer = []
    all_new_steer = []
    all_new_throttle = []
    all_new_brake = []
    all_new_command = []

    for episode in os.listdir(INPUT_DIR):
        in_episode_path = os.path.join(INPUT_DIR, episode)
        if not os.path.isdir(in_episode_path):
            continue

        csv_file = os.path.join(in_episode_path, "controls_nav.csv")
        if not os.path.exists(csv_file):
            print(f"Sari peste {episode} (Nu exista controls_nav.csv)")
            continue
        
        rows = []
        header = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                continue
            for line in reader:
                rows.append(line)
        
        if not rows: continue

        steerings = [float(row[1]) for row in rows]
        all_original_steer.extend(steerings)
        
        smoothed_steerings = smooth_steering(steerings)

        out_episode_path = os.path.join(OUTPUT_DIR, episode)
        os.makedirs(out_episode_path)

        new_rows = []
        for i, row in enumerate(rows):
            img_name = row[0]
            new_steer = smoothed_steerings[i] 
            throttle = float(row[2]) 
            brake = float(row[3])    
            command = int(row[4]) 

            
            if abs(new_steer) < 0.05 and brake < 0.1:
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
            all_new_steer.append(new_steer)
            all_new_throttle.append(throttle)
            all_new_brake.append(brake)
            all_new_command.append(command)

            
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
                all_new_steer.append(flip_steer)
                all_new_throttle.append(throttle)
                all_new_brake.append(brake)
                all_new_command.append(flip_command)

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
    print(f"Total cadre finale (cu Flip): {total_kept}")
    
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Analiza Dataset-ului Procesat pentru Trafic", fontsize=16, fontweight='bold')

    #Grafic suprapunere Original vs Procesat
    axes[0, 0].hist(all_original_steer, bins=50, color='red', alpha=0.5, label='Original')
    axes[0, 0].hist(all_new_steer, bins=50, color='green', alpha=0.7, label='Procesat (Echilibrat)')
    axes[0, 0].set_title("1. Distribuția Volanului (Steer)")
    axes[0, 0].set_xlabel("Unghi Volan (-1.0 Stânga, 1.0 Dreapta)")
    axes[0, 0].set_ylabel("Număr de Cadre")
    axes[0, 0].legend()

    #Grafic Acceleratie
    axes[0, 1].hist(all_new_throttle, bins=20, color='blue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title("2. Accelerație (Throttle)")
    axes[0, 1].set_xlabel("Putere (0.0 Oprit, 1.0 Max)")
    axes[0, 1].set_ylabel("Număr de Cadre")

    #Grafic Frana
    axes[1, 0].hist(all_new_brake, bins=20, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title("3. Frânare (Brake)")
    axes[1, 0].set_xlabel("Putere Frână (0.0 Liber, 1.0 Max)")
    axes[1, 0].set_ylabel("Număr de Cadre")

    #Grafic Comenzi GPS
    cmd_labels = ['LANE (0)', 'LEFT (1)', 'RIGHT (2)', 'STRAIGHT (3)']
    cmd_counts = [all_new_command.count(i) for i in range(4)]
    bars = axes[1, 1].bar(cmd_labels, cmd_counts, color=['gray', 'purple', 'cyan', 'magenta'], edgecolor='black')
    axes[1, 1].set_title("4. Comenzi GPS")
    axes[1, 1].set_ylabel("Număr de Cadre")
    
    
    for bar in bars:
        yval = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2, yval + (max(cmd_counts)*0.01), int(yval), ha='center', va='bottom', fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()

if __name__ == "__main__":
    main()