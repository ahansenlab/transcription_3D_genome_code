import os
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
PARENT_DIRECTORY = '/mnt/hildeshareslocal/Jamie/processed/dCtcf'
QC_DIRECTORY = PARENT_DIRECTORY

# Output folder to save QC'ed tracks
OUTPUT_DIRECTORY = '/mnt/md1/Jamie/tracks/dCtcf'

def filter_and_copy_tracks():
    print(f"Source Directory: {PARENT_DIRECTORY}")
    print(f"Output Directory: {os.path.abspath(OUTPUT_DIRECTORY)}")
    print("-" * 60)
    
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

    if not os.path.exists(PARENT_DIRECTORY):
        print(f"Error: Parent directory not found at {PARENT_DIRECTORY}")
        return
        
    movies = [d for d in os.listdir(PARENT_DIRECTORY) if os.path.isdir(os.path.join(PARENT_DIRECTORY, d))]
    print(f"Found {len(movies)} movie folders to process.\n")

    for movie in tqdm(movies, desc="Processing Movies"):
        raw_movie_dir = os.path.join(PARENT_DIRECTORY, movie)
        qc_file_path = os.path.join(QC_DIRECTORY, movie, 'corrections', 'annotations.csv')
        out_movie_dir = os.path.join(OUTPUT_DIRECTORY, movie)
        
        if not os.path.exists(qc_file_path):
            continue
            
        try:
            df_eval = pd.read_csv(qc_file_path)
            
            accepted_eval = df_eval[df_eval['decision'] != 0].copy()
            
            if accepted_eval.empty:
                continue
                
            for _, row in accepted_eval.iterrows():
                track_filename = f"nuclei/{int(row['nucleus']):03d}/tracks/{int(row['track']):03d}.csv"
                out_track_filename = f"{int(row['nucleus']):03d}_{int(row['track']):03d}.csv"
                raw_track_path = os.path.join(raw_movie_dir, track_filename)
                
                if os.path.exists(raw_track_path):
                    df_track = pd.read_csv(raw_track_path)
                    
                    start_frame = row['start']
                    end_frame = row['end']
                    # df_filtered = df_track[(df_track['frame'] >= start_frame) & (df_track['frame'] <= end_frame)]
                    df_filtered = df_track.iloc[start_frame:end_frame+1]
                    
                    if not df_filtered.empty:
                        os.makedirs(out_movie_dir, exist_ok=True)
                        
                        out_track_path = os.path.join(out_movie_dir, out_track_filename)
                        df_filtered.to_csv(out_track_path, index=False)
                        
        except Exception as e:
            print(f"\n[!] Error processing movie {movie}: {e}")

    print(f"\nDone! Filtered tracks have been saved to: {os.path.abspath(OUTPUT_DIRECTORY)}")

if __name__ == "__main__":
    filter_and_copy_tracks()