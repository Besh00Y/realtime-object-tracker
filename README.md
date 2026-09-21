# Real-Time Object Tracker

This project is a real-time object tracker built with Python, OpenCV, and YOLOv8.

The idea is simple: when the program starts, the user chooses whether to use a webcam or a video file. In the first frame, the user selects the object they want to track using a bounding box. The system then finds the YOLO detection that best matches the selected area using IoU and keeps tracking that object using its tracking ID.

The processed video is also saved as `output_tracking.mp4`.

## Features

* Track a specific object selected by the user.
* Supports both webcam and video files.
* Manual object selection in the first frame.
* YOLOv8 for object detection and tracking.
* Uses IoU to match the selected object with a YOLO detection.
* Keeps track of the selected object using its tracking ID.
* Shows the object class and tracking ID.
* Shows `Target Lost` if the selected object is no longer detected.
* Saves the processed video to an MP4 file.
* Resizes the displayed output to make high-resolution videos easier to view.

## Project Structure

```text
realtime-object-tracker/
│
├── tracker.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How It Works

The tracker follows a few simple steps.

### 1. Choose the input source

The program supports two types of input:

* Webcam
* Video file

The source is selected when running the program using the `--source` argument.

For example, to use the webcam:

```bash
python tracker.py --source 0
```

For a video file:

```bash
python tracker.py --source videos/walking.mp4
```

The program checks whether the source can be opened before starting the tracking process.

### 2. Select the object

After the first frame is read, the program opens an OpenCV window called `Select Object`.

The user selects the object they want to track by drawing a bounding box around it.

The selected ROI is returned by OpenCV as:

```text
(x, y, width, height)
```

This is then converted to:

```text
(x1, y1, x2, y2)
```

so it can be compared with the YOLO bounding boxes.

### 3. Detect and track objects in the first frame

YOLOv8 is loaded using Ultralytics:

```python
model = YOLO("yolov8n.pt")
```

The first frame is passed to:

```python
model.track(
    first_frame,
    persist=True
)
```

Using `persist=True` allows the tracker to maintain object identities across frames.

### 4. Match the selected object using IoU

The manually selected bounding box is compared with all YOLO detections in the first frame.

The project uses Intersection over Union (IoU):

```text
IoU = Intersection Area / Union Area
```

The detection with the highest IoU is considered the selected target.

The IoU calculation is implemented manually in `utils.py`.

This keeps the calculation separate from the main tracking code.

### 5. Get the target tracking ID

After finding the YOLO detection with the highest IoU, the program gets its tracking ID.

For example:

```text
Target ID: 3
Best IoU: 0.72
```

From this point, the program looks for this ID in every following frame.

### 6. Track the selected object

For every new frame, YOLO runs tracking again.

The program checks all detected objects and looks for the same tracking ID.

If the target is found, its bounding box is drawn in green.

The displayed label contains the object class and its tracking ID:

```text
person | ID: 3
```

### 7. Target Lost

If the target ID is not found in the current frame, the program does not assign another object to the target.

Instead, it displays:

```text
Target Lost
```

This makes it clear that the selected object is currently not being tracked.

The tracker continues processing the following frames.

## Output Video

The processed frames are saved using OpenCV's `VideoWriter`.

The output file is:

```text
output_tracking.mp4
```

The output video is saved at:

```text
1280 x 720
```

The original frame is used for YOLO processing, while the frame is resized only before displaying and saving the result.

This is useful when the input video has a very high resolution such as:

```text
3840 x 2160
```

because displaying the original resolution can make the video window unnecessarily large.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/realtime-object-tracker.git
```

Then move into the project directory:

```bash
cd realtime-object-tracker
```

### 2. Create a virtual environment

On Windows:

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### 3. Install the required packages

```bash
pip install -r requirements.txt
```

## Requirements

The main libraries used in this project are:

```text
ultralytics
opencv-python
```

The YOLOv8 model is downloaded automatically by Ultralytics when it is not already available locally.

## Running the Project

### Webcam

To use the default webcam:

```bash
python tracker.py --source 0
```

### Video File

To use a video file:

```bash
python tracker.py --source path/to/video.mp4
```

For example:

```bash
python tracker.py --source videos/walking.mp4
```

When the program starts:

1. The selected input source is opened.
2. The first frame is displayed.
3. Select the object you want to track.
4. Confirm the selection.
5. The system finds the closest YOLO detection using IoU.
6. The target ID is assigned.
7. The object is tracked through the following frames.
8. The processed result is displayed and saved.

## Controls

During object selection:

* Use the mouse to draw a bounding box around the target.
* Press `Enter` or `Space` to confirm the selection.

During tracking:

* Press `Q` to stop the program.

## Example Output

When the target is being tracked:

```text
person | ID: 2
```

If the target disappears from the tracker:

```text
Target Lost
```

The processed video is saved as:

```text
output_tracking.mp4
```

## Implementation Details

The main tracking logic is implemented in `tracker.py`.

The IoU calculation is separated into `utils.py`:

```python
from utils import calculate_iou
```

The basic tracking pipeline is:

```text
Input Video / Webcam
        ↓
Read First Frame
        ↓
User Selects Object
        ↓
YOLOv8 Detection + Tracking
        ↓
Calculate IoU
        ↓
Find Matching Detection
        ↓
Get Target ID
        ↓
Track Target in Following Frames
        ↓
 ┌─────────────────────┐
 │ Target Found?       │
 └─────────┬───────────┘
       Yes │       No
           │        │
           ↓        ↓
     Draw Box    Target Lost
           │        │
           └────┬───┘
                ↓
          Display + Save
```

## Limitations

The current implementation depends on the object IDs generated by the YOLO tracker.

If the target is heavily occluded or the tracker loses its identity, the system may display `Target Lost`.

Objects that look very similar can also make tracking more challenging.

The current version does not attempt to re-identify the target after its tracking ID changes.

## Possible Improvements

Some possible future improvements include:

* Recovering the target after an ID change.
* Better handling of long-term occlusion.
* Configurable IoU thresholds.
* FPS monitoring.
* Using larger YOLO models for better detection accuracy.
* GPU acceleration.
* Adding a confidence threshold.
* Adding a target re-identification method.

## Author

**Beshoy Ashraf**

AI & Machine Learning Engineer
