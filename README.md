# Dataset Utility Tools

This repository contains utility tools to help with dataset preparation and management for machine learning projects.

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv

# Activate the virtual environment
# On Windows:
 venv\Scripts\activate
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

### 3. Image Labeler (`image_labeler.py`)

This all-in-one tool helps you create and manage image datasets with custom property labels. It can automatically detect properties using a lightweight CLIP model or allow manual annotation.

#### Usage:

```bash
# Manual annotation mode
python image_labeler.py /path/to/images/folder --output output.xlsx --fields field1 field2 field3

# Automatic annotation mode
python image_labeler.py /path/to/images/folder --output output.xlsx --fields field1 field2 field3 --auto
```

#### Parameters:

- `folder`: Path to the folder containing images
- `--output` or `-o`: Path to the output Excel file
- `--fields` or `-f`: Custom fields to add as columns (e.g., "wears_belt" "has_collar")
- `--auto` or `-a`: Use AI to automatically detect properties
- `--update` or `-u`: Update existing sheet instead of creating a new one
- `--batch` or `-b`: Batch size for processing images (default: 10)

#### Annotation Values:

- **1**: Property exists in the image
- **-1**: Property doesn't exist in the image
- **0**: Uncertain or not yet annotated (default for manual mode)

#### Examples:

```bash
# Create a new annotation sheet for manual labeling
python image_labeler.py ./dog_photos --output dog_annotations.xlsx --fields wears_belt has_collar color breed

# Automatically analyze images for specific properties
python image_labeler.py ./dog_photos --output dog_annotations.xlsx --fields wears_belt has_collar --auto

# Update existing annotation sheet with new images
python image_labeler.py ./new_dog_photos --output dog_annotations.xlsx --update

# Analyze a large dataset with smaller batch size
python image_labeler.py ./large_image_collection --output annotations.xlsx --fields has_person is_outdoor --auto --batch 5
```

## Requirements

- Python 3.6 or higher
- pandas
- openpyxl
- Pillow
- torch
- transformers

See `requirements.txt` for specific version requirements.

## License

[Your license information]

## Contributing

[Your contribution guidelines]
