# Player Detection and Re-Identification in Video using YOLO and Appearance Matching

## 1. About the Project

This project focuses on detecting and consistently tracking **players** in a sports video using a custom-trained YOLO model, without tracking the ball or other objects. The system is designed to **assign consistent player IDs**, even when players temporarily leave and reappear in the frame.

## 2. How to Setup and Run

### Setup

1. Clone the repository.
2. Place your input video as `input.mp4` in the project directory.
3. Ensure `yolov_model.pt` is present in the root directory (trained to detect "person").
4. Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Tracker

```bash
python main.py
```

The processed output will be saved as `output.mp4`.

## 3. Dependencies and Environment

- Python 3.8+
- OpenCV (`cv2`)
- Ultralytics YOLOv8
- NumPy

You can use a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 4. My Approach and Methodology

### Detection

- Uses YOLOv8 for detecting players.
- Filters low confidence and small area detections.
- Automatically selects the player/person class from the model.

### Tracking

- **IoU-based box matching** to track players frame-to-frame.
- **HSV histogram-based appearance re-identification** to match reappearing players.
- Maintains memory for `active_players` and `disappeared_players`.

### Visualization

- Draws bounding boxes and IDs.
- Displays frame number.
- Outputs annotated video as `output.mp4`.

## 5. Challenges in This Project

- Detecting only players, not the ball.
- Assigning consistent IDs despite occlusion or disappearance.
- Managing ID swaps when multiple players are close.
- Preventing ghost boxes due to incorrect reassignments.
- Ensuring real-time processing capability.

## 6. Techniques I Tried and Issues Faced

### Approach 1: IoU-Based Matching Only

- Simple matching via Intersection over Union.
- **Issue**: Assigned new IDs on reappearance. No memory of past players.

### Approach 2: IoU + Disappearance Buffer

- Introduced buffer to keep missing players in memory.
- **Issue**: Ball still detected as player. Re-identification weak.

### Approach 3: IoU + Shape Filtering

- Added box size and aspect ratio checks.
- **Issue**: Reduced ball detections. But IDs still swapped often.

### Approach 4: IoU + Histogram Appearance Matching

- Added HSV histograms + Bhattacharyya distance.
- **Issue**: Better re-ID. Slow in crowded scenes. ID swaps remained.

### Final (Current) Approach

- Combines: IoU + HSV Histogram + Player Memory
- **Current Issues**:
  - ID swaps when players overlap.
  - Ghost tracks persist after players leave.

## 7. What Remains & Future Approaches

- **Kalman Filter + Hungarian Algorithm**: For better prediction and data association.
- **Use of deep embeddings**: Replace HSV histograms with deep appearance features.
- **Optical flow**: Enhance tracking with motion continuity.
- **Confidence decay mechanism**: Remove ghost boxes over time.
- **Trajectory validation**: Reject implausible ID reassignments.
- **Real-time optimization**: Consider TensorRT and multiprocessing.
- **Advanced dashboard**: Track positions, heatmaps, and player stats.

## Project Structure

```
.
├── main.py              # Final working tracker
├── input.mp4            # Input video (your own)
├── output.mp4           # Output video with tracking
├── yolov_model.pt       # YOLO model (custom-trained or pretrained)
├── requirements.txt     # Required Python packages
├── README.md            # This file
```

## Developer Info

This project is built and maintained by **Sambhavi Sharma** using OpenCV and Ultralytics YOLOv8.

- GitHub: [Sambhavi374](https://github.com/Sambhavi374)
- Email: [sambhavi374@gmail.com](mailto\:sambhavi374@gmail.com)

