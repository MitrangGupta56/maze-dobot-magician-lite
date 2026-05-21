import cv2
import numpy as np
import os
import json

# ---------------------------
# Configuration
# ---------------------------
CAM_INDEX = 2

# Output paths
SAVE_PATH = "/home/mdg/Documents/RS1/midsem2/rectified_maze.jpg"
MAZE_DATA_PATH = "/home/mdg/Documents/RS1/midsem2/transform_data.json"

# ---------------------------
# Circle Detection (Red/Green markers)
# ---------------------------
def detect_colored_circles(img_bgr):
    """Detect red and green circles using HSV color filtering"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    circles = {}
    
    # Red: two ranges (wraps around in HSV)
    red_mask1 = cv2.inRange(hsv, np.array([0, 100, 80]), np.array([10, 255, 255]))
    red_mask2 = cv2.inRange(hsv, np.array([170, 100, 80]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    
    # Green
    green_mask = cv2.inRange(hsv, np.array([35, 100, 80]), np.array([85, 255, 255]))
    
    for mask, color_name, color_bgr in [(red_mask, 'red', (0, 0, 255)), 
                                         (green_mask, 'green', (0, 255, 0))]:
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_circle = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if 200 < area < 50000:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                if circularity > 0.6 and 8 < radius < 100:
                    score = circularity * area
                    
                    if score > best_score:
                        best_score = score
                        best_circle = {
                            'center': (int(x), int(y)),
                            'radius': int(radius),
                            'color': color_name,
                            'color_bgr': color_bgr
                        }
        
        if best_circle:
            circles[color_name] = best_circle
    
    return circles


# ---------------------------
# Maze Boundary Detection
# ---------------------------
def detect_maze_bounds(img_bgr):
    """Detect the outer boundary of the white maze"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold: white maze = 255, black background = 0
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Check if we need to invert
    if (binary > 0).mean() > 0.9:
        binary = 255 - binary
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get largest contour (should be maze boundary)
    largest = max(contours, key=cv2.contourArea)
    
    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)
    
    bounds = {
        'top_left': (x, y),
        'top_right': (x + w, y),
        'bottom_right': (x + w, y + h),
        'bottom_left': (x, y + h),
        'width': w,
        'height': h
    }
    
    return bounds


# ---------------------------
# IMPROVED: Create High-Quality Binary Maze Mask
# ---------------------------
def create_walkable_mask(img_bgr, bounds):
    """Create accurate binary mask: 255=walkable (white), 0=wall (black)"""
    if not bounds:
        return None
    
    h, w = img_bgr.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better wall detection
    # This handles lighting variations better than global threshold
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Also try Otsu for comparison
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combine both methods (AND operation = stricter wall detection)
    binary_combined = cv2.bitwise_and(binary, binary_otsu)
    
    # Clean up noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_clean = cv2.morphologyEx(binary_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary_clean = cv2.morphologyEx(binary_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create mask and restrict to maze bounds
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y = bounds['top_left']
    maze_w, maze_h = bounds['width'], bounds['height']
    mask[y:y+maze_h, x:x+maze_w] = binary_clean[y:y+maze_h, x:x+maze_w]
    
    return mask


# ---------------------------
# Wall and Path Extraction
# ---------------------------
def extract_maze_structure(img_bgr, bounds):
    """Extract wall (black) and path (white) information within maze bounds"""
    if not bounds:
        return None
    
    # Get high-quality binary mask
    binary = create_walkable_mask(img_bgr, bounds)
    
    if binary is None:
        return None
    
    # Crop to maze area
    x, y = bounds['top_left']
    w, h = bounds['width'], bounds['height']
    binary_crop = binary[y:y+h, x:x+w]
    
    # Find wall pixels (black = 0) and path pixels (white = 255)
    wall_coords = np.column_stack(np.where(binary_crop == 0))
    path_coords = np.column_stack(np.where(binary_crop == 255))
    
    # Convert back to original image coordinates
    wall_coords_global = [(int(x + pt[1]), int(y + pt[0])) for pt in wall_coords]
    path_coords_global = [(int(x + pt[1]), int(y + pt[0])) for pt in path_coords]
    
    # Sample points (too many pixels to store all)
    sample_rate = max(1, len(wall_coords_global) // 5000)
    wall_sample = wall_coords_global[::sample_rate]
    
    sample_rate = max(1, len(path_coords_global) // 5000)
    path_sample = path_coords_global[::sample_rate]
    
    return {
        'walls': wall_sample,
        'paths': path_sample,
        'wall_count': len(wall_coords_global),
        'path_count': len(path_coords_global)
    }


# ---------------------------
# Main Processing Function
# ---------------------------
def process_frame(frame_bgr):
    """Process frame and extract all maze data"""
    
    # Detect circles
    circles_dict = detect_colored_circles(frame_bgr)
    
    # Detect maze bounds
    bounds = detect_maze_bounds(frame_bgr)
    
    # Extract maze structure
    maze_structure = None
    walkable_mask = None
    if bounds:
        maze_structure = extract_maze_structure(frame_bgr, bounds)
        walkable_mask = create_walkable_mask(frame_bgr, bounds)
    
    # Create visualization
    output = frame_bgr.copy()
    
    # Draw circles
    for circle in circles_dict.values():
        cv2.circle(output, circle['center'], circle['radius'], circle['color_bgr'], 2)
        cv2.circle(output, circle['center'], 3, circle['color_bgr'], -1)
        cv2.putText(output, circle['color'], 
                   (circle['center'][0] + circle['radius'] + 5, circle['center'][1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, circle['color_bgr'], 2)
    
    # Draw maze bounds
    if bounds:
        tl = bounds['top_left']
        tr = bounds['top_right']
        br = bounds['bottom_right']
        bl = bounds['bottom_left']
        
        cv2.line(output, tl, tr, (0, 255, 255), 2)
        cv2.line(output, tr, br, (0, 255, 255), 2)
        cv2.line(output, br, bl, (0, 255, 255), 2)
        cv2.line(output, bl, tl, (0, 255, 255), 2)
        
        for pt, label in [(tl, 'TL'), (tr, 'TR'), (br, 'BR'), (bl, 'BL')]:
            cv2.circle(output, pt, 8, (255, 255, 0), -1)
            cv2.putText(output, label, (pt[0] + 12, pt[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    
    # Prepare data output
    h, w = frame_bgr.shape[:2]
    
    circles_list = []
    for color, circle in circles_dict.items():
        circles_list.append({
            'center': circle['center'],
            'color': color
        })
    
    maze_data = {
        'image_size': {'width': w, 'height': h},
        'circles': circles_list,
        'maze_bounds': bounds,
        'maze_structure': maze_structure,
        'detection_status': {
            'circles_found': len(circles_dict),
            'bounds_found': bounds is not None,
            'structure_extracted': maze_structure is not None
        }
    }
    
    # Status text
    status_parts = []
    status_parts.append(f"Circles: {len(circles_dict)}/2")
    status_parts.append("Maze: " + ("✓" if bounds else "✗"))
    
    if maze_structure:
        status_parts.append(f"Walls: {maze_structure['wall_count']}")
        status_parts.append(f"Paths: {maze_structure['path_count']}")
    
    status = " | ".join(status_parts)
    
    cv2.putText(output, status, (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
               (0, 255, 0) if len(circles_dict) == 2 and bounds else (0, 165, 255), 2)
    
    ready = len(circles_dict) == 2 and bounds is not None
    
    # Also save the walkable mask for debugging
    maze_data['walkable_mask_available'] = walkable_mask is not None
    
    return output, frame_bgr, maze_data, ready, walkable_mask


# ---------------------------
# Main Program
# ---------------------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {CAM_INDEX}")
        print("Try changing CAM_INDEX (0, 1, 2, etc.)")
        return
    
    cv2.namedWindow("Camera - Raw Maze Detection", cv2.WINDOW_NORMAL)
    
    print("\n" + "="*60)
    print("RAW MAZE DETECTION (Improved Wall Detection)")
    print("="*60)
    print("\nControls:")
    print("  SPACE/C - Capture and save maze")
    print("  Q/ESC   - Quit")
    print("\nDetection Requirements:")
    print("  • White maze on dark background")
    print("  • Red circle visible")
    print("  • Green circle visible")
    print("  • Camera positioned steady")
    print("  • Good lighting (no shadows on walls)")
    print("\n" + "="*60 + "\n")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame")
            break
        
        # Process frame
        output, raw_frame, maze_data, ready, walkable_mask = process_frame(frame)
        
        # Display
        cv2.imshow("Camera - Raw Maze Detection", output)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key in (27, ord('q')):  # ESC or Q
            print("\nQuitting...")
            break
        
        elif key in (32, ord('c')):  # SPACE or C
            if ready:
                os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
                
                # Save raw image
                success = cv2.imwrite(SAVE_PATH, raw_frame)
                
                if success:
                    # Save walkable mask for solver to use
                    if walkable_mask is not None:
                        mask_path = SAVE_PATH.replace('.jpg', '_mask.png')
                        cv2.imwrite(mask_path, walkable_mask)
                        maze_data['walkable_mask_path'] = mask_path
                    
                    # Save maze data JSON
                    with open(MAZE_DATA_PATH, 'w') as f:
                        json.dump(maze_data, f, indent=2)
                    
                    print("\n" + "="*60)
                    print("✓ CAPTURE SUCCESSFUL")
                    print("="*60)
                    print(f"  Image: {SAVE_PATH}")
                    print(f"  JSON:  {MAZE_DATA_PATH}")
                    if walkable_mask is not None:
                        print(f"  Mask:  {mask_path}")
                    print(f"  Size:  {maze_data['image_size']['width']}x{maze_data['image_size']['height']}")
                    print(f"  Circles: {maze_data['detection_status']['circles_found']}/2")
                    
                    if maze_data['maze_structure']:
                        print(f"  Walls: {maze_data['maze_structure']['wall_count']} pixels")
                        print(f"  Paths: {maze_data['maze_structure']['path_count']} pixels")
                    
                    print("="*60 + "\n")
                    
                    break
                else:
                    print("\n✗ Failed to save image\n")
            else:
                missing = []
                if maze_data['detection_status']['circles_found'] < 2:
                    missing.append("both circles")
                if not maze_data['detection_status']['bounds_found']:
                    missing.append("maze boundary")
                
                print(f"\n✗ Cannot capture - missing: {', '.join(missing)}\n")
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
