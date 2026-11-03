import pandas as pd
import numpy as np

def generate_bulk_dataset(records=1000):
    data = []
    
    for _ in range(records):
        # Randomly assign 30% of the bulk data as "Dengue Positive" patterns
        is_dengue = np.random.choice([0, 1], p=[0.7, 0.3])
        
        if is_dengue:
            # Typical Dengue Pattern
            wbc = np.random.uniform(2000, 4800)       # Leukopenia (Low)
            platelets = np.random.uniform(15000, 95000) # Thrombocytopenia (Very Low)
            hct = np.random.uniform(46, 55)            # Hemoconcentration (High)
            hb = np.random.uniform(14.5, 17.5)         # High Hemoglobin
        else:
            # Typical Healthy/Other Infection Pattern
            wbc = np.random.uniform(5000, 11000)      # Normal
            platelets = np.random.uniform(155000, 450000) # Normal
            hct = np.random.uniform(38, 44)            # Normal
            hb = np.random.uniform(12.0, 14.5)         # Normal
            
        data.append([round(wbc, 0), round(platelets, 0), round(hct, 1), round(hb, 1), is_dengue])

    df = pd.DataFrame(data, columns=['WBC', 'Platelets', 'Hematocrit', 'Hemoglobin', 'Result'])
    
    # Save as CSV for Batch Upload testing
    df.to_csv('test_dengue_data.csv', index=False)
    print(f"Successfully generated 'test_dengue_data.csv' with {records} records.")

if __name__ == "__main__":
    generate_bulk_dataset(1000)