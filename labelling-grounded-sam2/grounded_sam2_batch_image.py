import os

# Set environment variables before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor 
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
import json
import argparse

def process_images_in_folder(
    input_folder,
    output_folder,
    json_output_folder,
    text_prompt,
    grounding_model,
    image_predictor,
    box_threshold=0.25,
    text_threshold=0.25
):
    """Process all JPG images in a folder and create annotations"""
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder does not exist: {input_folder}")
    
    # Create output directories
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(json_output_folder, exist_ok=True)
    
    # Get all JPG files
    image_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.jpg', '.jpeg'))
    ]
    image_files.sort()
    
    if not image_files:
        print(f"No JPG/JPEG files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, image_file)
        
        try:
            # Load image
            image_source, image = load_image(img_path)
            
            # Predict with Grounding DINO
            boxes, confidences, labels = predict(
                model=grounding_model,
                image=image,
                caption=text_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
            
            # Skip if no detections
            if len(boxes) == 0:
                print(f"No detections found in {image_file}")
                # Create empty JSON file
                frame_data = {
                    "filename": image_file,
                    "image_width": image_source.shape[1],
                    "image_height": image_source.shape[0],
                    "detections": []
                }
                json_filename = os.path.splitext(image_file)[0] + '.json'
                json_path = os.path.join(json_output_folder, json_filename)
                with open(json_path, 'w') as f:
                    json.dump(frame_data, f, indent=2)
                continue
            
            # Process boxes for SAM 2 - ENSURE PROPER DATA TYPES
            h, w, _ = image_source.shape
            
            # Make sure boxes is a proper tensor and move to CPU for processing
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes.copy())  # Copy to make it writable
            
            # Ensure boxes is float32
            boxes = boxes.float()
            
            # Scale boxes properly
            boxes_scaled = boxes * torch.tensor([w, h, w, h], dtype=torch.float32)
            input_boxes = box_convert(boxes=boxes_scaled, in_fmt="cxcywh", out_fmt="xyxy")
            
            # Ensure numpy arrays are writable and correct dtype
            input_boxes = input_boxes.detach().cpu().numpy().copy().astype(np.float32)
            confidences_np = confidences.detach().cpu().numpy().copy().astype(np.float32)
            
            # Set image for SAM predictor
            image_predictor.set_image(image_source)
            
            # Get masks from SAM 2
            masks, scores, logits = image_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )
            
            # Ensure proper mask format
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            
            # Create detections for visualization - fix indexing
            class_ids = np.arange(len(labels)).astype(np.int32)  # Ensure integer type
            
            detections = sv.Detections(
                xyxy=input_boxes.astype(np.float32),
                mask=masks.astype(bool),
                confidence=confidences_np.astype(np.float32),
                class_id=class_ids
            )
            
            # Annotate image
            img_copy = image_source.copy()
            
            # Add bounding boxes
            box_annotator = sv.BoxAnnotator()
            img_copy = box_annotator.annotate(scene=img_copy, detections=detections)
            
            # Add labels
            label_annotator = sv.LabelAnnotator()
            img_copy = label_annotator.annotate(
                scene=img_copy, 
                detections=detections, 
                labels=[f"{label} {conf:.2f}" for label, conf in zip(labels, confidences_np)]
            )
            
            # Add masks
            mask_annotator = sv.MaskAnnotator()
            img_copy = mask_annotator.annotate(scene=img_copy, detections=detections)
            
            # Save annotated image
            output_filename = f"annotated_{image_file}"
            output_path = os.path.join(output_folder, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
            
            # Create JSON label file
            frame_data = {
                "filename": image_file,
                "image_width": int(w),
                "image_height": int(h),
                "detections": []
            }
            
            for i, (box, label, conf) in enumerate(zip(input_boxes, labels, confidences_np)):
                detection = {
                    "class": str(label),
                    "confidence": float(conf),
                    "bbox": {
                        "xmin": float(box[0]),
                        "ymin": float(box[1]),
                        "xmax": float(box[2]),
                        "ymax": float(box[3])
                    }
                }
                frame_data["detections"].append(detection)
            
            # Save JSON file
            json_filename = os.path.splitext(image_file)[0] + '.json'
            json_path = os.path.join(json_output_folder, json_filename)
            with open(json_path, 'w') as f:
                json.dump(frame_data, f, indent=2)
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            import traceback
            traceback.print_exc()  # This will help debug the exact error
            continue

def main():
    parser = argparse.ArgumentParser(description="Process images in a folder with Grounded SAM 2")
    parser.add_argument("--input_folder", required=True, help="Input folder containing JPG images")
    parser.add_argument("--output_folder", required=True, help="Output folder for annotated images")
    parser.add_argument("--json_output_folder", required=True, help="Output folder for JSON label files")
    parser.add_argument("--text_prompt", required=True, help="Text prompt for Grounding DINO")
    parser.add_argument("--box_threshold", type=float, default=0.25, help="Box threshold for Grounding DINO")
    parser.add_argument("--text_threshold", type=float, default=0.25, help="Text threshold for Grounding DINO")
    parser.add_argument("--grounding_config", default="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    parser.add_argument("--grounding_checkpoint", default="gdino_checkpoints/groundingdino_swint_ogc.pth")
    parser.add_argument("--sam2_checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    
    args = parser.parse_args()
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load Grounding DINO model
    print("Loading Grounding DINO model...")
    grounding_model = load_model(
        model_config_path=args.grounding_config,
        model_checkpoint_path=args.grounding_checkpoint,
        device=device
    )
    
    # Load SAM 2 model
    print("Loading SAM 2 model...")
    sam2_model = build_sam2(args.sam2_config, args.sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_model)
    
    # Enable optimizations for newer GPUs
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Process images
    process_images_in_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        json_output_folder=args.json_output_folder,
        text_prompt=args.text_prompt,
        grounding_model=grounding_model,
        image_predictor=image_predictor,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold
    )
    
    print("Processing complete!")

if __name__ == "__main__":
    main()