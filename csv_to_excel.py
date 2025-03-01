import os
import argparse
import pandas as pd
from pathlib import Path

def convert_csv_to_excel(csv_path, output_path=None):
    csv_file = Path(csv_path)
    
    if not csv_file.exists():
        print(f"Error: File '{csv_path}' does not exist")
        return False
    
    if not csv_file.is_file():
        print(f"Error: '{csv_path}' is not a file")
        return False
    
    if csv_file.suffix.lower() != '.csv':
        print(f"Warning: '{csv_path}' does not have a .csv extension")
    
    if output_path is None:
        output_file = csv_file.with_suffix('.xlsx')
    else:
        output_file = Path(output_path)
    
    try:
        print(f"Reading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        print(f"Converting to Excel: {output_file}")
        df.to_excel(output_file, index=False)
        
        print(f"Conversion successful! Excel file saved at: {output_file}")
        return True
        
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV file: {e}")
    except PermissionError:
        print(f"Permission denied when writing to {output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
    
    return False

def main():
    parser = argparse.ArgumentParser(description='Convert CSV files to Excel format')
    parser.add_argument('csv_file', help='Path to the input CSV file')
    parser.add_argument('--output', '-o', help='Path to the output Excel file (optional)')
    
    args = parser.parse_args()
    
    convert_csv_to_excel(args.csv_file, args.output)

if __name__ == "__main__":
    main()
