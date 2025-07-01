# GuardianAI: Old-Age Care Action Detection

GuardianAI is an AI-powered system for real-time detection and analysis of human actions (e.g., falls, waving) using advanced computer vision and deep learning. It is designed to assist in monitoring elderly individuals for safety and well-being.

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```sh
git clone https://github.com/FVLegion/GuardianAI.git
cd inference
```

### 2. Create and Activate a Virtual Environment

```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

**Key libraries:**
- `torch`, `torchvision` – Deep learning backend
- `ultralytics` – YOLOv8-pose detection (add to requirements.txt if used)
- `opencv-python` – Video capture and visualization
- `matplotlib`, `numpy` – Plotting and numerical operations
- `pygame`, `sounddevice` – Audio/alert support
- `streamlit` – Web interface
- `clearml` – Experiment tracking and pipeline automation

> **Note:** Python 3.11.9 is recommended for optimal compatibility.

---

## 💻 Compatibility Notes

✅ Python 3.11.9 recommended ONLY
✅ GPU support optional but improves performance (CUDA-compatible) minimum 4GB VRAM 
❗ Tested on Windows 11, MAC OS
🖥️ Requires minimum 4GB RAM and a dual-core CPU

---

## 🧾 Configuration

Edit `bytetrack.yaml` to customize system behavior:

```yaml
tracker_type: bytetrack      # tracker type, ['botsort', 'bytetrack']
track_high_thresh: 0.25      # threshold for the first association
track_low_thresh: 0.1        # threshold for the second association
new_track_thresh: 0.25       # threshold for new track if the detection does not match tracks
track_buffer: 120            # buffer to calculate the time when to remove tracks
match_thresh: 1              # threshold for matching tracks
fuse_score: True             # Whether to fuse confidence scores with the iou distances
```

✅ Ensure the camera source is properly connected before launch.

- **Model paths** can be set in the UI or via environment variables:
  - `POSE_MODEL_PATH` (default: `models/yolo11m-pose.pt`)
  - `ACTION_MODEL_PATH` (default: `models/BiLSTMWithAttention_best_model_3.0.pth`)

---

## 🚀 Running the Application

### Streamlit UI (Recommended)

```sh
streamlit run inference/streamlit3.py
```

### Command-Line Scripts

- **Single Person Demo:**  
  ```sh
  streamlit run inference/great_solo.py
  ```
- **Multi-Person Demo Local:**  
  ```sh
  streamlit run inference/great_multi.py
  ```
- **Multi-Person Demo Streamlit**  
  ```sh
  streamlit run inference/streamlit3.py
  ```

---

## 🧪 Model Checkpoint

Ensure that the pretrained model weights are present in the `models/` directory:

- `models/BiLSTMWithAttention_best_model_3.0.pth` (Action recognition)
- `models/yolo11m-pose.pt` (YOLOv11m pose detection)

If missing, download from the link provided by the development team or request access.

---

## 📁 Project Structure

```
GuardianAI/
│
├── clearML/
│   ├── GuardianPipeline.py
│   ├── datasetEDA.py
│   ├── model.py
│   ├── pose_dataset.py
│   └── ...
├── inference/
│   ├── streamlit3.py
│   ├── great_solo.py
│   ├── great_multi.py
│   ├── Great.py
│   ├── pose.py
│   ├── model.py
│   └── ...
├── models/
│   ├── BiLSTMWithAttention_best_model_3.0.pth
│   ├── yolo11n-pose.pt
│   └── ...
├── requirements.txt
└── README.md
```

---

## ⚙️ ClearML Pipeline

- The ClearML pipeline is defined in [`clearML/GuardianPipeline.py`](clearML/GuardianPipeline.py).
- To run the pipeline, configure your ClearML credentials in `clearML/FVLEGION.txt` or `clearML/KongML.txt`.

---

## 📝 Notes

- **Camera Source:** Ensure your webcam or IP camera is connected and accessible.
- **Audio Alerts:** Place your alert sound files in the `inference/sounds/` directory.
- **Custom Models:** You can upload custom YOLO or action recognition models via the Streamlit sidebar.

---

## 📢 Support

For issues, please open an issue on GitHub or contact the development team.

---

## 👥 Contributors

This project was developed by a team of dedicated students from the University of Technology Sydney (UTS):

### Team Members

- **Abdullah Ibrahim Bahri** ([@the777bahri](https://github.com/the777bahri))
  - Student ID: 13384589
  - Email: Abdullah.i.bahri@student.uts.edu.au

- **Hayrambh Monga** ([@Hayrambh](https://github.com/orgs/FVLegion/people/Hayrambh))
  - Student ID: 25176811
  - Email: hayrambhrajesh.monga@student.uts.edu.au

- **Shudarshan Singh Kongkham** ([@ShudarshanKongkham](https://github.com/orgs/FVLegion/people/ShudarshanKongkham))
  - Student ID: 25024199
  - Email: shudarshan.s.kongkham@student.uts.edu.au

We appreciate the collaborative effort and dedication each team member brought to making GuardianAI a reality.

---

## 📢 Support

// ... existing code ...
---
