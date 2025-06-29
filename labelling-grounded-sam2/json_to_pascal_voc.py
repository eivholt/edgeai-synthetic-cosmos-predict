import os
import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse

def create_pascal_voc_xml(json_data, image_filename, output_path):
    """Convert JSON detection data to Pascal VOC XML"""
    
    # Create root element
    annotation = ET.Element("annotation")
    
    # Add basic info
    folder = ET.SubElement(annotation, "folder")
    folder.text = "images"
    
    filename = ET.SubElement(annotation, "filename")
    filename.text = image_filename
    
    # Add size
    size = ET.SubElement(annotation, "size")
    width = ET.SubElement(size, "width")
    width.text = str(json_data["image_width"])
    height = ET.SubElement(size, "height")
    height.text = str(json_data["image_height"])
    depth = ET.SubElement(size, "depth")
    depth.text = "3"
    
    segmented = ET.SubElement(annotation, "segmented")
    segmented.text = "0"
    
    # Add objects
    for detection in json_data["detections"]:
        obj = ET.SubElement(annotation, "object")
        
        name = ET.SubElement(obj, "name")
        name.text = detection["class"]
        
        pose = ET.SubElement(obj, "pose")
        pose.text = "Unspecified"
        
        truncated = ET.SubElement(obj, "truncated")
        truncated.text = "0"
        
        difficult = ET.SubElement(obj, "difficult")
        difficult.text = "0"
        
        bndbox = ET.SubElement(obj, "bndbox")
        xmin = ET.SubElement(bndbox, "xmin")
        xmin.text = str(int(detection["bbox"]["xmin"]))
        ymin = ET.SubElement(bndbox, "ymin")
        ymin.text = str(int(detection["bbox"]["ymin"]))
        xmax = ET.SubElement(bndbox, "xmax")
        xmax.text = str(int(detection["bbox"]["xmax"]))
        ymax = ET.SubElement(bndbox, "ymax")
        ymax.text = str(int(detection["bbox"]["ymax"]))
    
    # Create pretty XML
    rough_string = ET.tostring(annotation, 'unicode')
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Clean up XML
    lines = [line for line in pretty_xml.split('\n') if line.strip()]
    pretty_xml = '\n'.join(lines)
    
    with open(output_path, 'w') as f:
        f.write(pretty_xml)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", required=True, help="Directory containing JSON files")
    parser.add_argument("--xml_dir", required=True, help="Output directory for XML files")
    parser.add_argument("--image_ext", default=".jpg", help="Image file extension")
    args = parser.parse_args()
    
    os.makedirs(args.xml_dir, exist_ok=True)
    
    for json_file in os.listdir(args.json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(args.json_dir, json_file)
            
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Create corresponding image filename
            frame_name = json_file.replace('.json', args.image_ext)
            xml_name = json_file.replace('.json', '.xml')
            xml_path = os.path.join(args.xml_dir, xml_name)
            
            create_pascal_voc_xml(data, frame_name, xml_path)
            print(f"Converted {json_file} -> {xml_name}")