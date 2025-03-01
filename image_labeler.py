import os
import argparse
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import requests
from io import BytesIO
import sys
import time
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import pipeline

class AdvancedImageAnalyzer:
    """Class to analyze images using specialized Vision Transformer models."""
    
    def __init__(self):
        """Initialize the image analyzer with specialized models."""
        print("Initializing Advanced Image Analyzer...")
        
        try:
            # Load a specialized image classification model
            self.processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
            self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
            
            # Load object detection pipeline for more precise property detection
            self.object_detector = pipeline(
                "object-detection", 
                model="facebook/detr-resnet-50", 
                threshold=0.2  # Lowered threshold for better detection
            )
            
            # Move model to GPU if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)
            
            # Initialize zero-shot classification for custom property detection
            try:
                self.zero_shot = pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-large-patch14", 
                    device=0 if torch.cuda.is_available() else -1
                )
            except Exception:
                print("Falling back to standard CLIP model...")
                self.zero_shot = pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-base-patch32",
                    device=0 if torch.cuda.is_available() else -1
                )
            
            print(f"Models loaded successfully on {self.device}!")
        except Exception as e:
            print(f"Error loading models: {e}")
            try:
                # Fallback to a simpler model if the first attempt fails
                print("Attempting to load fallback model...")
                self.zero_shot = pipeline(
                    "zero-shot-image-classification",
                    model="openai/clip-vit-base-patch32",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("Fallback model loaded successfully!")
            except Exception as e2:
                print(f"Error loading fallback model: {e2}")
                sys.exit(1)
        
        # Define common property descriptions for better detection
        self.property_templates = {
            # Clothing items
            "belt": ["person wearing belt", "belt around waist", "clothing with belt", "belt buckle visible", 
                     "leather belt", "waist with belt", "pants with belt", "visible belt"],
            "collar": ["shirt with collar", "dress collar", "collar on clothing", "visible collar", 
                       "collar detail", "collar around neck", "standing collar", "folded collar"],
            "hat": ["person wearing hat", "hat on head", "baseball cap", "sun hat", "cowboy hat"],
            "glasses": ["person wearing glasses", "eyeglasses", "sunglasses on face", "spectacles"],
            "tie": ["person wearing tie", "necktie", "bow tie", "tie around neck"],
            "scarf": ["person wearing scarf", "scarf around neck", "neck scarf", "winter scarf"],
            "jacket": ["person wearing jacket", "coat", "outerwear", "zippered jacket"],
            "gloves": ["hands with gloves", "wearing gloves", "winter gloves", "work gloves"],
            
            # Default fallback for unknown properties
            "default": ["contains {}", "has {}", "shows {}", "with {}", "visible {}"]
        }

    def get_property_descriptions(self, property_name):
        """Get specialized descriptions for common properties."""
        property_lower = property_name.lower()
        
        # Check for common clothing items by looking at substrings
        for key, templates in self.property_templates.items():
            if key in property_lower or property_lower in key:
                return templates
        
        # If no specific match, use the default templates with property name inserted
        return [template.format(property_name) for template in self.property_templates["default"]]

    def analyze_image(self, image_path, property_name):
        """Analyze if a specific property exists in the image using multiple detection methods.
        
        Args:
            image_path: Path to the image file
            property_name: Name of the property to check for
            
        Returns:
            int: 1 (property exists), -1 (property doesn't exist), or 0 (uncertain)
        """
        try:
            # Load image
            if isinstance(image_path, str) and image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
                
            # Get specialized property descriptions
            property_descriptions = self.get_property_descriptions(property_name)
            
            # Create negative descriptions
            negative_descriptions = [
                f"without {property_name}",
                f"no {property_name}",
                f"missing {property_name}",
                f"lacks {property_name}"
            ]
            
            # Add "person without belt" type descriptions for clothing items
            if property_name.lower() in ["belt", "collar", "hat", "glasses", "tie", "scarf", "jacket", "gloves"]:
                negative_descriptions.extend([
                    f"person without {property_name}",
                    f"outfit without {property_name}",
                    f"clothing without {property_name}"
                ])
            
            # Adjust the candidate labels based on property name
            candidate_labels = property_descriptions + negative_descriptions
                
            # Run zero-shot classification
            results = self.zero_shot(
                image, 
                candidate_labels=candidate_labels, 
                top_k=len(candidate_labels)
            )
            
            # Compute average score for positive and negative descriptions
            positive_scores = []
            negative_scores = []
            
            for item in results:
                # Check if this is a negative description
                is_negative = any(neg in item["label"].lower() for neg in ["without", "no ", "missing", "lacks"])
                
                if is_negative:
                    negative_scores.append(item["score"])
                else:
                    positive_scores.append(item["score"])
            
            # Check if we have scores to work with
            if not positive_scores or not negative_scores:
                return 0  # Uncertain if we don't have both types of scores
            
            # Calculate weighted score - give more weight to highest scores
            positive_scores.sort(reverse=True)
            negative_scores.sort(reverse=True)
            
            # Take top 3 scores if available
            top_positive = positive_scores[:min(3, len(positive_scores))]
            top_negative = negative_scores[:min(3, len(negative_scores))]
            
            avg_positive = sum(top_positive) / len(top_positive)
            avg_negative = sum(top_negative) / len(top_negative)
            
            # Debugging information
            print(f"    Top positive score: {max(positive_scores) if positive_scores else 0:.4f}")
            print(f"    Top negative score: {max(negative_scores) if negative_scores else 0:.4f}")
            
            # Set threshold for more decisive classification
            # Lower threshold to be more decisive for clothing items
            confidence_threshold = 0.08 if property_name.lower() in ["belt", "collar", "wears_belt", "has_collar"] else 0.1
            
            if avg_positive > avg_negative and (avg_positive - avg_negative) > confidence_threshold:
                return 1  # Property exists with high confidence
            elif avg_negative > avg_positive and (avg_negative - avg_positive) > confidence_threshold:
                return -1  # Property doesn't exist with high confidence
            
            # 2. Try object detection as a secondary method for common objects
            try:
                # Object detection can help for physical objects that can be detected
                objects = self.object_detector(image)
                
                # Check if any detected object matches or relates to the property
                property_keywords = property_name.lower().replace("_", " ").split()
                
                # Map common property names to related object detection labels
                property_mapping = {
                    "belt": ["belt", "waistband", "strap"],
                    "collar": ["collar", "shirt", "necklace", "tie"],
                    "wears_belt": ["belt", "waistband", "strap"],
                    "has_collar": ["collar", "shirt", "necklace", "tie"]
                }
                
                # Get related keywords for this property
                related_keywords = property_mapping.get(property_name.lower(), [])
                all_keywords = property_keywords + related_keywords
                
                # Check detected objects against our keywords
                for obj in objects:
                    obj_label = obj['label'].lower()
                    # Check if any keyword is in the detected object's label
                    if any(keyword in obj_label for keyword in all_keywords):
                        print(f"    Detected object: {obj_label} (score: {obj['score']:.2f})")
                        if obj['score'] > 0.3:  # Only count if reasonably confident
                            return 1  # Property exists based on object detection
                
                # If specific object detection didn't work, check for people
                # as a secondary signal for clothing items
                person_detected = any(obj['label'].lower() == 'person' for obj in objects)
                
                # If still uncertain from zero-shot but leaning positive
                if person_detected and avg_positive > avg_negative:
                    return 1  # Give benefit of doubt for clothing on detected people
                
                # If we've checked with object detection and found nothing related
                # AND the zero-shot classification was leaning negative
                if avg_negative > avg_positive:
                    return -1  # Property likely doesn't exist
            except Exception as obj_err:
                # If object detection fails, continue with other methods
                print(f"    Object detection skipped: {obj_err}")
            
            # 3. Make a final decision based on highest individual scores
            # This helps to break ties in uncertain cases
            if max(positive_scores) > max(negative_scores) + 0.05:
                return 1
            elif max(negative_scores) > max(positive_scores) + 0.05:
                return -1
                
            # If still uncertain, return 0
            return 0
                
        except Exception as e:
            print(f"Error analyzing image {image_path} for property {property_name}: {e}")
            return 0  # Return uncertain in case of error

def scan_images(folder_path):
    """Scan a folder for image files and return a list of image paths."""
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Error: '{folder_path}' is not a valid directory")
        return []
    
    image_files = []
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in image_extensions:
            image_files.append(file)
    
    image_files.sort()
    return image_files

def process_images(folder_path, output_excel, custom_fields, auto_detect=False, batch_size=10, update_mode=False):
    """Process images and create/update Excel sheet with annotations."""
    # Scan for images
    image_files = scan_images(folder_path)
    
    if not image_files:
        print(f"No image files found in '{folder_path}'")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Check if output Excel exists
    excel_file = Path(output_excel)
    excel_exists = excel_file.exists()
    
    # Initialize variables for update mode
    new_images = []
    new_fields = []
    
    # If updating existing file, read it first
    if update_mode and excel_exists:
        try:
            existing_df = pd.read_excel(output_excel)
            # Get list of images already processed
            existing_images = existing_df['image_filename'].tolist() if 'image_filename' in existing_df.columns else []
            
            # Filter out already processed images
            new_images = [img for img in image_files if img.name not in existing_images]
            
            if not new_images:
                print("All images are already in the Excel sheet.")
                if auto_detect and custom_fields:
                    print("Checking for new fields to analyze...")
                    # Check for new fields
                    new_fields = [field for field in custom_fields if field not in existing_df.columns]
                    if not new_fields:
                        print("No new fields to analyze.")
                        return True
                    else:
                        # We have new fields to process for existing images
                        print(f"Found {len(new_fields)} new fields to analyze for {len(existing_images)} images.")
                        # Map existing filenames back to file paths
                        image_files = []
                        for img_name in existing_images:
                            matching_files = [img for img in scan_images(folder_path) if img.name == img_name]
                            if matching_files:
                                image_files.append(matching_files[0])
                        custom_fields = new_fields
                else:
                    return True
            else:
                print(f"Found {len(new_images)} new images to process.")
                image_files = new_images
                
        except Exception as e:
            print(f"Error reading existing Excel file: {e}")
            excel_exists = False
            update_mode = False
    
    # Initialize the analyzer if auto_detect is True
    analyzer = None
    if auto_detect:
        analyzer = AdvancedImageAnalyzer()
        print("Using advanced image analyzer for automatic property detection")
    
    # Process images
    all_results = []
    
    # Process images in batches if auto_detect is True, otherwise process all at once
    if auto_detect:
        # Process in batches
        for i in range(0, len(image_files), batch_size):
            batch = image_files[i:i+batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(image_files)-1)//batch_size + 1} ({len(batch)} images)")
            
            for img in batch:
                img_path = str(img.absolute())
                img_name = img.name
                
                print(f"Analyzing image: {img_name}")
                
                # Create dictionary for this image's results
                img_data = {
                    'image_filename': img_name,
                    'image_path': img_path
                }
                
                # Analyze each property
                for field in custom_fields:
                    print(f"  Checking property: {field}")
                    result = analyzer.analyze_image(img_path, field)
                    img_data[field] = result
                    
                    # Print the result with more descriptive status
                    if result == 1:
                        status = "EXISTS ✓"
                    elif result == -1:
                        status = "ABSENT ✗"
                    else:
                        status = "UNCERTAIN ?"
                    print(f"  Result: Property '{field}' {status}")
                
                all_results.append(img_data)
                print(f"Completed analysis for {img_name}\n")
    else:
        # Create data structure without analysis
        for img in image_files:
            img_data = {
                'image_filename': img.name,
                'image_path': str(img.absolute())
            }
            
            # Add custom fields with default value 0 (uncertain)
            for field in custom_fields:
                img_data[field] = 0
            
            all_results.append(img_data)
    
    # Create DataFrame with results
    results_df = pd.DataFrame(all_results)
    
    # Save or update Excel file
    if update_mode and excel_exists:
        try:
            if len(new_fields) > 0 and len(new_images) == 0:
                # Only adding new fields to existing images
                for field in new_fields:
                    existing_df[field] = results_df[field].values
                existing_df.to_excel(output_excel, index=False)
            else:
                # Combine with existing data
                updated_df = pd.concat([existing_df, results_df], ignore_index=True)
                updated_df.to_excel(output_excel, index=False)
            print(f"Updated annotation sheet saved at: {output_excel}")
        except Exception as e:
            print(f"Error updating Excel file: {e}")
            return False
    else:
        try:
            results_df.to_excel(output_excel, index=False)
            print(f"New annotation sheet created at: {output_excel}")
        except Exception as e:
            print(f"Error creating Excel file: {e}")
            return False
    
    print("\nAnnotation Values Guide:")
    print("  1: Property exists in the image ✓")
    print(" -1: Property doesn't exist in the image ✗")
    print("  0: Uncertain/not annotated yet ?")
    
    # Print detection statistics if auto_detect was used
    if auto_detect and all_results:
        property_stats = {}
        for field in custom_fields:
            values = [item[field] for item in all_results if field in item]
            positive = values.count(1)
            negative = values.count(-1)
            uncertain = values.count(0)
            property_stats[field] = {
                "positive": positive,
                "negative": negative,
                "uncertain": uncertain,
                "total": len(values)
            }
        
        print("\nDetection Statistics:")
        for field, stats in property_stats.items():
            positive_percent = (stats["positive"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            negative_percent = (stats["negative"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            uncertain_percent = (stats["uncertain"] / stats["total"]) * 100 if stats["total"] > 0 else 0
            
            print(f"  {field}:")
            print(f"    Positive: {stats['positive']} ({positive_percent:.1f}%)")
            print(f"    Negative: {stats['negative']} ({negative_percent:.1f}%)")
            print(f"    Uncertain: {stats['uncertain']} ({uncertain_percent:.1f}%)")
    
    return True

def main():
    parser = argparse.ArgumentParser(description='Process and label images with custom properties')
    parser.add_argument('folder', help='Path to the folder containing images')
    parser.add_argument('--output', '-o', required=True, help='Path to the output Excel file')
    parser.add_argument('--fields', '-f', nargs='+', help='Custom fields to add as columns (e.g., "wears_belt" "has_collar")')
    parser.add_argument('--auto', '-a', action='store_true', help='Use AI to automatically detect properties')
    parser.add_argument('--update', '-u', action='store_true', help='Update existing sheet instead of creating a new one')
    parser.add_argument('--batch', '-b', type=int, default=10, help='Batch size for processing images (default: 10)')
    
    args = parser.parse_args()
    
    if not args.fields:
        print("Warning: No custom fields specified. Use --fields to add custom columns.")
        fields = []
    else:
        fields = args.fields
    
    process_images(
        args.folder, 
        args.output, 
        fields, 
        auto_detect=args.auto, 
        batch_size=args.batch, 
        update_mode=args.update
    )

if __name__ == "__main__":
    main()
