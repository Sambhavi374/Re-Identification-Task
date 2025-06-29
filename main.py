import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import time

# Configuration
IOU_THRESHOLD = 0.3
HIST_THRESHOLD = 0.5  # Increased threshold for better matching
MAX_UNSEEN_FRAMES = 30  # ~1 second at 30fps
MAX_MISSING_FRAMES = 150  # ~5 seconds at 30fps
MIN_BOX_AREA = 1000  # Increased min area to filter out small false positives
HIST_BINS = [16, 16]  # Fewer bins for faster processing
CONF_THRESHOLD = 0.5  # Higher confidence threshold to reduce false positives

# Load YOLO model
try:
    model = YOLO("yolov_model.pt")
    print("Model loaded successfully")
    
    # Get class names from model
    class_names = model.names
    print(f"Model classes: {class_names}")
    
    # Find player class ID
    player_class_id = None
    for idx, name in class_names.items():
        if 'player' in name.lower() or 'person' in name.lower():
            player_class_id = idx
            print(f"Using class index {player_class_id} for players")
            break
    
    if player_class_id is None:
        print("Warning: 'player' class not found in model. Using class 0 as fallback.")
        player_class_id = 0

except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize video
cap = cv2.VideoCapture("input.mp4")
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {w}x{h} at {fps:.2f} fps")

# Create output video
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Tracking state
active_players = {}  # {player_id: {box, hist, frames_unseen}}
disappeared_players = {}  # {player_id: {last_box, last_hist, frames_missing}}
next_id = 0


# Calculate Intersection over Union (IoU) between two boxes
def compute_iou(box1, box2):
    
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return inter_area / float(box1_area + box2_area - inter_area)


# Compute normalized HSV histogram for a player crop
def compute_histogram(crop):
    
    if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
        return None
        
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, HIST_BINS, [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


# Clamp box coordinates to valid image bounds
def clamp_box(box, width, height):
    
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width-1))
    y1 = max(0, min(int(y1), height-1))
    x2 = max(0, min(int(x2), width-1))
    y2 = max(0, min(int(y2), height-1))
    return [x1, y1, x2, y2]

# Main processing loop
frame_count = 0
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    print(f"\n--- Processing Frame {frame_count} ---")
    
    # Run object detection
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    
    # Process detections
    current_detections = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            
            # Only process player detections
            if cls_id == player_class_id:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                # Filter by minimum area
                if area < MIN_BOX_AREA:
                    continue
                    
                clamped_box = clamp_box([x1, y1, x2, y2], w, h)
                crop = frame[clamped_box[1]:clamped_box[3], clamped_box[0]:clamped_box[2]]
                
                if crop.size == 0:
                    continue
                    
                hist = compute_histogram(crop)
                if hist is not None:
                    current_detections.append({
                        'box': clamped_box,
                        'crop': crop,
                        'hist': hist
                    })
                    print(f"Detected player at {clamped_box} with conf {conf:.2f}")
    
    print(f"Found {len(current_detections)} valid player detections")
    
    # Initialize matching states
    matched_detections = [False] * len(current_detections)
    matched_players = {pid: False for pid in active_players}
    
    # Step 1: Match with active players using IoU (spatial matching)
    for d_idx, detection in enumerate(current_detections):
        best_iou = 0
        best_match = None
        
        for pid, player in active_players.items():
            if matched_players[pid]:
                continue
                
            iou = compute_iou(detection['box'], player['box'])
            if iou > best_iou:
                best_iou = iou
                best_match = pid
        
        if best_iou > IOU_THRESHOLD:
            # Update player state
            player = active_players[best_match]
            player['box'] = detection['box']
            player['hist'] = detection['hist']
            player['frames_unseen'] = 0
            matched_detections[d_idx] = True
            matched_players[best_match] = True
            print(f"Matched player {best_match} via IoU ({best_iou:.2f})")
    
    # Step 2: Match remaining detections using appearance features
    for d_idx, detection in enumerate(current_detections):
        if matched_detections[d_idx]:
            continue
            
        best_match = None
        best_similarity = float('inf')
        
        for pid, player in active_players.items():
            if matched_players[pid]:
                continue
                
            # Compare histograms using Bhattacharyya distance
            similarity = cv2.compareHist(
                detection['hist'], 
                player['hist'], 
                cv2.HISTCMP_BHATTACHARYYA
            )
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = pid
        
        # Only match if similarity is below threshold
        if best_match is not None and best_similarity < HIST_THRESHOLD:
            # Update player state
            player = active_players[best_match]
            player['box'] = detection['box']
            player['hist'] = detection['hist']
            player['frames_unseen'] = 0
            matched_detections[d_idx] = True
            matched_players[best_match] = True
            print(f"Matched player {best_match} via appearance (dist: {best_similarity:.3f})")
    
    # Step 3: Check disappeared players for reappearances
    for d_idx, detection in enumerate(current_detections):
        if matched_detections[d_idx]:
            continue
            
        best_match = None
        best_similarity = float('inf')
        
        for pid, player in disappeared_players.items():
            if player['frames_missing'] > MAX_MISSING_FRAMES:
                continue
                
            # Compare histograms
            similarity = cv2.compareHist(
                detection['hist'], 
                player['last_hist'], 
                cv2.HISTCMP_BHATTACHARYYA
            )
            
            if similarity < best_similarity:
                best_similarity = similarity
                best_match = pid
        
        # Only match if similarity is below threshold
        if best_match is not None and best_similarity < HIST_THRESHOLD:
            # Reactivate player
            active_players[best_match] = {
                'box': detection['box'],
                'hist': detection['hist'],
                'frames_unseen': 0
            }
            disappeared_players.pop(best_match)
            matched_detections[d_idx] = True
            print(f"Reidentified player {best_match} (dist: {best_similarity:.3f})")
    
    # Step 4: Create new players for unmatched detections
    for d_idx, detection in enumerate(current_detections):
        if matched_detections[d_idx]:
            continue
            
        active_players[next_id] = {
            'box': detection['box'],
            'hist': detection['hist'],
            'frames_unseen': 0
        }
        print(f"New player detected: ID {next_id}")
        next_id += 1
    
    # Step 5: Update active players (handle disappearances)
    for pid in list(active_players.keys()):
        if not matched_players.get(pid, False):
            active_players[pid]['frames_unseen'] += 1
            
            if active_players[pid]['frames_unseen'] > MAX_UNSEEN_FRAMES:
                # Move to disappeared state
                disappeared_players[pid] = {
                    'last_box': active_players[pid]['box'],
                    'last_hist': active_players[pid]['hist'],
                    'frames_missing': 0
                }
                del active_players[pid]
                print(f"Player {pid} marked as disappeared")
    
    # Step 6: Update disappeared players
    for pid in list(disappeared_players.keys()):
        disappeared_players[pid]['frames_missing'] += 1
        
        if disappeared_players[pid]['frames_missing'] > MAX_MISSING_FRAMES:
            del disappeared_players[pid]
            print(f"Player {pid} removed from memory")
    
    # Visualization - Only show active players
    for pid, player in active_players.items():
        x1, y1, x2, y2 = player['box']
        color = (0, 255, 0)  # Green for active players
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{pid}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Show frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Write frame to output
    out.write(frame)
    
    # Calculate processing speed
    if frame_count % 10 == 0:
        elapsed = time.time() - start_time
        fps = frame_count / elapsed
        print(f"Processing speed: {fps:.2f} FPS")


cap.release()
out.release()
print(f"\nProcessing complete. Total frames: {frame_count}")
