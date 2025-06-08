# 🚦 Smart Traffic Monitoring System with YOLOv8

A real-time computer vision application for traffic detection and congestion analysis using YOLOv8, OpenCV, and Python. Built with a modern GUI and designed for both live webcam feed and video file input.

---

# 📁 Project Structure

Traffic Monitoring using CV
├── Code → Python source files, YOLOv8 model, and configs
│ ├── main.py
│ ├── config.json
│ └── yolov8n.pt
|
├── Compiled → Standalone app build with YOLO model
│ └── app.exe (or .app/.bin depending on OS)
├
|── Data 
│ ├── Input → Raw video files for processing
│ └── Output → Processed videos with detection overlays
├
|── README.md
└── requirements.txt

# 🔍 Features

- **Live Camera Feed**: Real-time traffic monitoring via webcam  
- **Video Upload & Analysis**: Analyze pre-recorded videos  
- **Vehicle Detection**: Uses YOLOv8 for real-time detection and counting  
- **Congestion Analysis**: Measures traffic density dynamically  
- **User Interface**: Clean, dark-themed GUI built with Tkinter  
- **Real-Time Metrics**: Live dashboard for traffic statistics  
- **Compiled App**: One-click executable for direct use without setup


# Requirements

- Python 3.8 or higher
- OpenCV
- YOLOv8
- Tkinter (usually comes with Python)
- Ultralytics

# Installation

1. Clone this repository:

- git clone [repository-url]
- cd traffic-monitoring

2. Install the required packages:

- pip install -r requirements.txt

# Usage

1. Run the application:
- main.app

2. The main interface provides four primary options:

- **Main Display**: Shows the current video feed with vehicle detection overlays
- **Metrics Display**: Shows real-time traffic statistics
- **overlay heatmap**: overlay a heatmap on the video feed
- **Replay Output**: Review previously analyzed footage  
- **Use Camera**: Start real-time analysis using your computer's camera 
- **Upload Video**: Select a pre-recorded video file for analysis

3. During analysis, the application will display:

   - Live analysis video feed with vehicle detection
   - Traffic density measurements
   - Congestion status


# Technical Details

- Uses YOLOv8 neural network for vehicle detection
- Implements OpenCV for video processing
- Built with Tkinter for the graphical user interface
- Supports multiple video formats (MP4 recommended)
- Real-time frame processing and analysis

# Limitations

- Camera functionality requires a connected webcam
- Performance depends on your system's hardware capabilities specifically the CPU as this is un optimized for GPU 
- YOLOv8 model accuracy may vary in different lighting conditions

# Troubleshooting

1. If the camera doesn't work:
   - Check if your webcam is properly connected
   - Ensure no other application is using the camera
   - Verify camera permissions for the application

2. If video analysis is slow:
   - Consider using a smaller resolution video
   - Ensure your system meets the minimum requirements
   - Close other resource-intensive applications

## License
 This project is licensed under the MIT License.
