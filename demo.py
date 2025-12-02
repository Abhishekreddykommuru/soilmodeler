import pandas as pd
import glob
import sys
import io

print("--- Starting File Inspection Script (v2) ---")
print("Trying to read as CSV first, as files might be misnamed.")

# Find all files to process
files_to_inspect = glob.glob('spectra_with_target_T*.xls')

if not files_to_inspect:
    print("No 'spectra_with_target_T*.xls' files found.")
    sys.exit() # Exit if no files

print(f"Found {len(files_to_inspect)} files to inspect.\n")

for file_path in files_to_inspect:
    print(f"=========================================")
    print(f"Inspecting: {file_path}")
    print(f"=========================================")
    
    df = None
    
    # --- STRATEGY 1: Try reading as a CSV ---
    # This is the most likely case given the 'BOF' error.
    try:
        print("Attempt 1: Reading as CSV...")
        # We read the file as raw bytes first to handle potential encoding issues
        with open(file_path, 'rb') as f:
            content = f.read()
        
        # The error showed '410,431,' which implies a comma delimiter
        # We'll use StringIO to treat the byte content as a file
        df = pd.read_csv(io.StringIO(content.decode('utf-8')), sep=',')
        print("...Success! File appears to be a CSV.")

    except Exception as e_csv:
        print(f"...CSV read failed: {e_csv}")
        
        # --- STRATEGY 2: Fallback to Excel (xlrd) ---
        if df is None:
            print("Attempt 2: Reading as Excel (xlrd)...")
            try:
                df = pd.read_excel(file_path, engine='xlrd')
                print("...Success! File is a valid Excel .xls file.")
            except Exception as e_xls:
                print(f"...Excel read failed: {e_xls}")
                print(f"\n*** FAILED to read {file_path} with all methods ***")
                continue # Skip to the next file

    # --- If we successfully loaded the dataframe (df) ---
    if df is not None:
        print("\n--- First 5 Rows (head) ---")
        print(df.head())
        
        print("\n--- Column List ---")
        print(list(df.columns))
        
        print(f"\nSuccessfully read {file_path}.\n")

print("--- Inspection Finished ---")