# Bundesliga Computer Vision Annotation Project

### Overview

This project explores how computer vision techniques and machine learning models—particularly the YOLO family—can be used to analyze football (soccer) match footage. By taking a 30-second Bundesliga scouting video, the pipeline detects, tracks, and annotates players, referees, and the ball, estimates ball possession, and identifies camera movement on the x-axis. The final output is an annotated video highlighting all these elements.

This project is largely an exploration on this [Main Video Resource](https://www.youtube.com/watch?v=neBZ6huolkg&t=5839s), yet it also includes general improvements/optimization that are reflected on the results. 

---

## Table of Contents

1. [Abstract](#abstract)
2. [Methodology](#methodology)
3. [Folder / File Structure](#folder--file-structure)
4. [Limitations](#limitations)
5. [Practical Applications](#practical-applications)
6. [Conclusion](#conclusion)
7. [Citations](#citations)

---

## Abstract

Machine learning is booming, and football (soccer) has not been left behind. This project leverages **computer vision** models to analyze 30-second clips from the German Bundesliga. Taking inspiration from a Kaggle competition, we detect and track players, referees, and the ball, identify camera movement, and approximate team possession—all within an annotated video output.

---

## Methodology

1. **Dockerized Environment & GPU Acceleration**:
    - A custom Docker-based environment ensures all dependencies are encapsulated and reproducible.
    - NVIDIA GPU acceleration speeds up both **model training** and **inference** steps.
    - Development and debugging carried out in **Visual Studio Code** for consistency.

1. **Data Collection & Preparation**:
    - Original video data was taken from a Kaggle competition and stored externally (Google Drive).
    - A labeled dataset from [Roboflow](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1) was used to train YOLO-based models (YOLOv5l, YOLOv5x, and YOLOv8x).
    - **Note**: The original dataset is not included in this repository due to licensing and storage limitations.

2. **Object Detection & Tracking**:
    - **YOLO (You Only Look Once)** models (v5 and v8) were used to detect objects (players, referees, ball).
    - **ByteTrack** was used to maintain consistent tracking IDs across frames, preventing sudden ID changes.

1. **Video Analysis & Annotation**:
    - The input video is split into frames.
    - Each frame is fed to the trained model to detect bounding boxes for all relevant objects.
    - **Ball Interpolation** fills in missing ball tracking data using Pandas interpolation to smooth transitions.
    - **Team Possession** is computed by checking which player is within a given distance threshold of the ball.
    - **Camera Movement** is estimated by analyzing feature points and tracking horizontal shifts (left or right).

1. **Output**:
    - All annotated frames (with bounding boxes, team color overlays, ball markers, possession details, and camera movement direction) are reassembled into an output video.
    - Various model versions (YOLOv5l, YOLOv5x, YOLOv8x) are tested and results compared.

---

## Key Files:

- **tracker.py**  
	Handles bounding box detection/tracking, interpolation for ball position, drawing overlays (ovals, triangles, possession stats, camera direction), and optimizing memory usage by processing frames one by one.
- **team_picker.py**  
	Classifies players into teams using jersey color clustering with KMeans. Special handling is included for goalkeepers.
- **camera_movement_estimator.py**  
	Estimates horizontal camera movement using optical flow, providing directional feedback in the final annotations.
- **player_ball_assigner.py** 
	Contains the logic for the player possession function. Identifies which *player* is closest to the ball in a given frame. 
- **team_possession.py**
	Identifies which *team* has possession of the ball. This team possession is ultimately calculated as a whole and is the final result annotated. 
- **main.py:**
	The main distributer for all the files. This file selects which model and video to use, calls all functions.

---

## Limitations

1. **Dataset Size**:
    - Only ~650 labeled images were used, which limits the model’s ability to generalize.
    - Combining multiple datasets or labeling more images could yield more accurate and robust models.

1. **Computational Resources**:
    - Training locally proved impractical due to long run times.
    - **Google Colab** was used for final training—sufficient for smaller-scale projects but not easily scalable for larger, production-level tasks unless Pro version is purchased.

---

## Practical Applications

1. **Sports Broadcasting**:
    - Live detection/annotation of players, ball tracking, and movement analysis for advanced in-game insights.
    - Potential use in _VAR_-like systems to detect offside lines and other critical match events.

1. **Data Analytics & Journalism**:
    - Adds a layer of intelligence to game analyses, enabling data-driven articles with possession metrics, passing networks, heat maps, etc.

1. **Beyond Football**:
    - Retail: foot traffic measurement and inventory management with real-time camera feeds.
    - Healthcare: automated anomaly detection in radiology scans.
    - Many other fields where real-time object detection and tracking can be invaluable.

---

## Conclusion

With machine learning’s continued growth, this project showcases how to take a small dataset and still produce meaningful, real-world insights. From tracking players and referees to interpolating the ball’s trajectory and estimating possession, the workflow demonstrates the power of **YOLO** models combined with robust data processing strategies. While the dataset and computational resources impose clear constraints, the resulting annotations and proof-of-concept indicate the potential for broader applications, especially if given more robust data and infrastructure.

---

## Citations

- [Roboflow Football Players Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc/dataset/1)
- [Kaggle DFL Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout)
- [Roboflow Supervision Documentation](https://supervision.roboflow.com/latest/)
- [Ultralytics YOLO GitHub](https://github.com/ultralytics/ultralytics)
- [Kaggle Discussion Thread on Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/359932)
- **YouTube Tutorials**
	- - [Main Video Resource](https://www.youtube.com/watch?v=neBZ6huolkg&t=5839s)
    - [Video 2](https://www.youtube.com/watch?v=aBVGKoNZQUw&t=108s)
    - [Video 3](https://www.youtube.com/watch?v=yJWAtr3kvPU&t=129s)
    - [Video 4](https://www.youtube.com/watch?v=ag3DLKsl2vk)