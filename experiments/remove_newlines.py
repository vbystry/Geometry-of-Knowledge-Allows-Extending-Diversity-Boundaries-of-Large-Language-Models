#!/usr/bin/env python3
"""
Script to remove newlines from CSV file fields.
Processes all CSV files in the current directory.
"""

import csv
import sys
import glob
import os

def remove_newlines_from_csv(input_file, output_file=None):
    """
    Remove newlines from all fields in a CSV file.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file (if None, overwrites input)
    """
    if output_file is None:
        output_file = input_file
    
    # Read the CSV file
    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        rows.append(header)
        
        # Process each row, removing newlines from all fields
        for row in reader:
            cleaned_row = [field.replace('\n', ' ').replace('\r', '') for field in row]
            rows.append(cleaned_row)
    
    # Write the cleaned CSV file
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"Successfully removed newlines from {input_file}")
    print(f"Output written to {output_file}")
    print(f"Processed {len(rows) - 1} data rows")
    print()

if __name__ == '__main__':
    # Find all CSV files in the current directory
    csv_files = glob.glob('*.csv')
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        sys.exit(1)
    
    print(f"Found {len(csv_files)} CSV file(s) to process:\n")
    for csv_file in sorted(csv_files):
        print(f"  - {csv_file}")
    print()
    
    # Process each CSV file
    for csv_file in sorted(csv_files):
        try:
            remove_newlines_from_csv(csv_file)
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            print()
    
    print("All CSV files processed!")



