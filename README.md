# Tennis Analyzer

The Tennis Analyzer program is a tool for analyzing tennis match videos, tracking players and the ball, and generating detailed statistics about player performance. It uses computer vision techniques to detect players and the ball, track their movements, and compute relevant statistics such as shot speeds and player speeds.

## Features

- **Player and Ball Tracking:** Detects players and the ball in video frames using YOLO models.
- **Court Line Detection:** Identifies court lines and keypoints using a trained model.
- **Mini Court Visualization:** Converts and visualizes detections in a mini court format.
- **Statistics Calculation:** Computes player and ball statistics, including shot speeds and player speeds.
- **Video Output:** Annotates the video with detected objects, statistics, and court lines, and saves the result to a new video file.

### Could not push the models (file size too large)


<img width="524" alt="image" src="https://github.com/user-attachments/assets/08ca84e4-1a6e-417b-a3ec-eb715cbf0de1">


## Code Overview

- **`utils.py`:** Contains utility functions for reading and saving videos, measuring distances, drawing player stats, and converting distances.
- **`constants.py`:** Defines constants used throughout the program.
- **`trackers.py`:** Contains classes for tracking players and the ball using YOLO models.
- **`court_line_detector.py`:** Contains the class for detecting court lines.
- **`mini_court.py`:** Provides functions for converting and visualizing detections on a mini court.

## Example

The program processes a tennis match video to detect players and the ball, calculates shot speeds and player speeds, and annotates the video with these statistics. The output is a video file where you can see the detected objects, court lines, and player statistics
