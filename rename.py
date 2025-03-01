import os
import argparse
from pathlib import Path

def rename_images(folder_path, base_name):
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory")
        return
    
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    image_files.sort()
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return
    
    print(f"Found {len(image_files)} image files")
    
    for i, file in enumerate(image_files, 1):
        new_name = f"{base_name}-{i:02d}{file.suffix}"
        new_path = folder / new_name
        
        if new_path.exists():
            print(f"Warning: '{new_name}' already exists, skipping")
            continue
        
        try:
            file.rename(new_path)
            print(f"Renamed: {file.name} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {file.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description='Rename image files in a folder with custom naming pattern')
    parser.add_argument('folder', help='Path to the folder containing images')
    parser.add_argument('--name', '-n', default='image', help='Base name for renamed images (default: "image")')
    
    args = parser.parse_args()
    
    rename_images(args.folder, args.name)
    
if __name__ == "__main__":
    main()