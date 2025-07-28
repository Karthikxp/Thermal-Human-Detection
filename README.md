# Thermal Human Detection for Drone & Surveillance Applications

This project utilizes a state-of-the-art YOLOv8 model to perform real-time human detection in thermal imagery. This capability is crucial for applications where standard visual cameras are ineffective, such as in complete darkness, through smoke and fog, or when subjects are visually camouflaged. By processing thermal video streams, this tool provides a powerful solution for security, surveillance, and rescue operations.

The core of the program is a YOLOv8 model fine-tuned for detecting human heat signatures, making it highly effective for deployment on drones and other remote platforms.

## Key Features

- **High-Accuracy Thermal Detection:** Leverages a pre-trained YOLOv8 model (`pitangent-ds/YOLOv8-human-detection-thermal`) specifically fine-tuned for human detection in thermal spectra.
- **Real-Time Performance:** Optimized for processing live video streams with low latency, suitable for immediate decision-making.
- **Flexible Video Input:** Easily configurable to use a standard webcam, a pre-recorded video file, or an RTSP stream from a drone or IP camera.
- **Clear Visualizations:** Uses the `supervision` library to draw clean, readable bounding boxes and confidence labels on detected individuals.

## Use Cases in Drone Technology

Drones equipped with thermal cameras and this detection software unlock a wide range of strategic capabilities across several domains.

### Military & Surveillance

- **Covert Night Operations:** Drones can operate stealthily in complete darkness, providing critical intelligence on enemy positions and movements without being detected.
- **Perimeter Security:** Automate the monitoring of perimeters for military bases, forward operating bases (FOBs), or other sensitive sites. The system can reliably detect intrusions in all weather and light conditions, day or night.
- **Force Protection & Overwatch:** A drone can fly over a friendly patrol, providing a "guardian angel" view that identifies potential ambushes or hidden threats (e.g., snipers) from a safe distance.
- **Counter-Camouflage:** Heat signatures cannot be hidden by traditional camouflage. This technology makes it trivial to spot individuals who are visually concealed in foliage, shadows, or ghillie suits.

### Disaster Management & Search and Rescue (SAR)

- **Locating Survivors:** Following natural disasters like earthquakes, floods, or building collapses, drones can rapidly scan vast and inaccessible areas. The thermal camera can detect the body heat of survivors who may be trapped under rubble, hidden by debris, or unconscious.
- **Firefighting Support:** In structure fires or wildfires, smoke obscures vision for both ground crews and standard cameras. Thermal drones can see directly through the smoke to locate trapped civilians or downed firefighters, guiding rescue teams to their exact location.
- **Wildfire Management:** Track the location of firefighting crews on the ground, identify dangerous hotspots, and monitor the fire's perimeter, even at night or in smoky conditions.

### Law Enforcement

- **Fugitive Searches:** Quickly search large areas, such as forests or rural landscapes, for fleeing suspects, especially during nighttime hours.
- **Crowd Monitoring:** Monitor large public gatherings for security threats or to manage crowd flow, ensuring public safety in low-light environments.

## How It Works

1.  **Model Loading:** The script downloads and initializes a `YOLO` object with the specialized thermal model from the Hugging Face Hub.
2.  **Video Capture:** It captures video frames from the configured source (default is the primary webcam).
3.  **Inference:** Each frame is passed to the YOLO model, which performs inference and returns a list of detections, including bounding box coordinates, class name (`person`), and a confidence score.
4.  **Annotation:** The `supervision` library is used to process these raw detections. It creates annotators that draw bounding boxes and formatted labels onto a copy of the frame.
5.  **Display:** The final annotated frame is displayed in a window, providing a real-time view of the detected individuals. The stream can be closed by pressing the `ESC` key.

## Setup & Installation

To get this project running on your local machine, follow these steps.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/thermal-pro.git
    cd thermal-pro
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    This project requires a few key Python libraries. You can install them using pip:
    ```bash
    pip install ultralytics supervision opencv-python huggingface_hub
    ```

## Usage

1.  **Run the script:**
    By default, the script will use your system's default webcam.
    ```bash
    python ml.py
    ```

2.  **Using a Different Video Source:**
    Open `ml.py` and modify the `cv2.VideoCapture()` line:
    -   **For a video file:**
        ```python
        cap = cv2.VideoCapture("path/to/your/video.mp4")
        ```
    -   **For an RTSP stream (from a drone or IP camera):**
        ```python
        cap = cv2.VideoCapture("rtsp://user:pass@ip_address:port/stream_path")
        ``` 