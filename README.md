# Dataset Utility Tools

This repository contains utility tools to help with dataset preparation and management for machine learning projects.

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
# venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install the required dependencies
pip install -r requirements.txt
```

## Available Tools

### 1. Image Renaming Tool (`rename.py`)

This tool allows you to batch rename image files in a folder with a consistent naming pattern.

#### Usage:

```bash
python rename.py /path/to/images/folder --name custom-name
```

#### Parameters:

- `folder`: Path to the folder containing images
- `--name` or `-n`: Base name for renamed images (default: "image")

#### Example:

```bash
# Rename all images in the 'dog_photos' folder to 'dog-01.jpg', 'dog-02.jpg', etc.
python rename.py ./dog_photos --name dog
```

### 2. CSV to Excel Converter (`csv_to_excel.py`)

This tool converts CSV files to Excel format, which can be useful for data preparation in machine learning workflows.

#### Usage:

```bash
python csv_to_excel.py /path/to/data.csv --output /path/to/output.xlsx
```

#### Parameters:

- `csv_file`: Path to the input CSV file
- `--output` or `-o`: Path to the output Excel file (optional, defaults to same name with .xlsx extension)

#### Example:

```bash
# Convert train_data.csv to Excel format
python csv_to_excel.py ./data/train_data.csv

# Convert with specific output path
python csv_to_excel.py ./data/train_data.csv --output ./processed/train_data.xlsx
```

## Requirements

- Python 3.6 or higher
- pandas
- openpyxl

See `requirements.txt` for specific version requirements.

## License

[Your license information]

## Contributing

[Your contribution guidelines]
