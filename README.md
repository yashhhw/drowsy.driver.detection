# Driver Drowsiness Detection System

## Overview

This project implements a real-time Driver Drowsiness Detection System using computer vision and facial recognition techniques. The system analyzes the driver's facial features to determine their alertness levels, aiming to reduce the risk of accidents caused by drowsiness.

## Features

- **Real-Time Drowsiness Detection**: Uses webcam input to monitor the driver's eyes and mouth for signs of drowsiness.
- **Eye Aspect Ratio (EAR)**: Calculates the EAR to detect when the driver's eyes are closed.
- **Mouth Aspect Ratio (MAR)**: Monitors the MAR to identify yawning behavior.
- **Score System**: Provides a score based on the driver's alertness level, triggering warnings when drowsiness is detected.
- **Facial Landmark Detection**: Highlights key facial points for visualization.

## Installation

To set up the project, ensure you have Python installed on your machine, and then install the required libraries:

## Usage

1. Connect a webcam to your system.
2. Run the main script to start the drowsiness detection.
3. The system will display a window with real-time feedback on the driver's alertness.

## Requirements

- Python 3.x
- OpenCV
- face_recognition
- NumPy
- Pillow
- SciPy
- Matplotlib

## Contributing

Feel free to fork the repository, make changes, and submit pull requests. Any contributions are welcome!

