import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
torch.classes.__path__ = []
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import tempfile
from PIL import Image
import io
import threading
import queue
import pygame
pygame.mixer.init()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device 💪: {device}")
# Add these globals or put them in your session state
frame_queue = queue.Queue(maxsize=2)
stop_event = threading.Event()

def frame_grabber(cap, frame_queue, stop_event):
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                frame_queue.get_nowait()  # Drop oldest frame
            except queue.Empty:
                pass
            frame_queue.put_nowait(frame)


# Set page config
st.set_page_config(page_title="Guardian(AI)👁️", layout="wide")

# ------------------------------------------------------------------------------
# 1. VISUALIZATION FUNCTIONS
# ------------------------------------------------------------------------------

# Keypoint connections (COCO format)
def get_keypoint_connections():
    return [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Nose, eyes, ears
        (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Shoulders, elbows, wrists
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16),  # Hips, knees, ankles
        (5, 11), (6, 12)  # Connect shoulders to hips
    ]

# Function to convert matplotlib figure to image
def plot_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    return img

# ------------------------------------------------------------------------------
# 2. DATA PROCESSING CLASSES
# ------------------------------------------------------------------------------

class PoseDataset(object):
    def __init__(self, data_dir, action_classes, max_frames=35):
        self.data_dir = data_dir
        self.action_classes = action_classes
        self.max_frames = max_frames
        self.feature_size = 34  # Expected feature size (17 keypoints * 2 coords)

    def process_keypoints(self, keypoints_data):
        all_frames_keypoints = []
        previous_frame_normalized = None  # Store the *normalized* previous frame for EMA
        alpha = 0.8  # Smoothing factor for EMA
        conf_threshold = 0.2

        # Ensure we only process up to max_frames from the end
        relevant_keypoints_data = keypoints_data[-self.max_frames:]

        for frame_data in relevant_keypoints_data:
            processed_frame = np.zeros(self.feature_size)  # Default to zeros for this frame
            try:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    all_frames_keypoints.append(processed_frame)  # Append zeros
                    continue

                frame_keypoints = frame_data['keypoints']
                # Check if keypoints list is valid and contains data for at least one person
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0 or \
                   not isinstance(frame_keypoints[0], list) or len(frame_keypoints[0]) == 0:
                    all_frames_keypoints.append(processed_frame)  # Append zeros
                    continue

                # Convert first person's keypoints, ensure shape (17, 3)
                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    all_frames_keypoints.append(processed_frame)  # Append zeros
                    continue

                # Filter out keypoints with low confidence for normalization stats
                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > conf_threshold]
                if valid_keypoints.shape[0] < 2:  # Need at least 2 points for mean/std
                    all_frames_keypoints.append(processed_frame)  # Append zeros
                    continue

                # Z-Score Normalization for X and Y based on valid points
                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8  # Add epsilon
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8  # Add epsilon

                # Apply normalization to all keypoints (using stats from valid ones)
                normalized_frame_keypoints = frame_keypoints_np.copy()
                normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

                # Temporal Smoothing using Exponential Moving Average (EMA) on normalized data
                if previous_frame_normalized is not None:
                    normalized_frame_keypoints[:, :2] = alpha * normalized_frame_keypoints[:, :2] + (1 - alpha) * previous_frame_normalized[:, :2]

                # Store the smoothed, normalized frame for the next iteration's EMA
                previous_frame_normalized = normalized_frame_keypoints

                # Flatten (X, Y coords only) and store
                processed_frame = normalized_frame_keypoints[:, :2].flatten()
                if processed_frame.shape[0] != self.feature_size:
                    processed_frame = np.zeros(self.feature_size)  # Fallback to zeros

            except Exception as e:
                processed_frame = np.zeros(self.feature_size)  # Append zeros on error

            all_frames_keypoints.append(processed_frame)

        # Padding (Pad at the BEGINNING)
        num_processed = len(all_frames_keypoints)
        if num_processed == 0:
            return None  # Indicate failure if absolutely no frames worked

        # Create the padded array
        padded_keypoints = np.zeros((self.max_frames, self.feature_size))

        # Calculate start index for copying data to pad at the beginning
        start_idx_pad = max(0, self.max_frames - num_processed)

        # Copy the processed frames into the padded array
        for i in range(num_processed):
            # Double check shape just in case
            if all_frames_keypoints[i].shape[0] == self.feature_size:
                padded_keypoints[start_idx_pad + i, :] = all_frames_keypoints[i]
            else:
                # This shouldn't happen often due to checks above, but as a fallback
                padded_keypoints[start_idx_pad + i, :] = np.zeros(self.feature_size)

        # Final shape check
        if padded_keypoints.shape != (self.max_frames, self.feature_size):
            return None  # Or handle error appropriately

        return padded_keypoints

# ------------------------------------------------------------------------------
# 3. MODEL ARCHITECTURE
# ------------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1)  # hidden_size * 2 for BiLSTM

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size * 2)
        scores = torch.tanh(self.attention_weights(lstm_output))  # Apply tanh for better gradient flow
        attention_weights = torch.softmax(scores, dim=1)  # Normalize scores to weights
        # Context vector: weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_size * 2)
        return context_vector, attention_weights.squeeze(-1)  # Return context and weights (batch_size, seq_length)


class ActionRecognitionBiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(ActionRecognitionBiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = AttentionLayer(hidden_size)
        # Input to FC is context vector size (hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)  # Apply dropout before the final layer

    def forward(self, x):
        # Initialize hidden and cell states (batch_size needs to be inferred from x)
        # Shape: (num_layers * num_directions, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # LSTM output: (batch_size, seq_length, hidden_size * 2)
        lstm_out, _ = self.lstm(x, (h0, c0))
        # Apply dropout to LSTM outputs before attention
        lstm_out = self.dropout(lstm_out)

        # Attention mechanism
        context_vector, attention_weights = self.attention(lstm_out)  # context_vector: (batch_size, hidden_size * 2)

        # Final classification layer
        # Apply dropout to context vector before FC layer
        context_vector_dropped = self.dropout(context_vector)
        logits = self.fc(context_vector_dropped)  # logits: (batch_size, num_classes)

        # Apply Softmax to get probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Return probabilities and attention weights
        return probabilities, attention_weights

# ------------------------------------------------------------------------------
# 4. PREDICTION FUNCTION
# ------------------------------------------------------------------------------

def predict_action(model, keypoints_sequence, action_classes, device, sequence_length, input_size):
    model.eval()  # Ensure model is in eval mode
    pose_dataset = PoseDataset("data", action_classes, max_frames=sequence_length)
    normalized_keypoints = pose_dataset.process_keypoints(keypoints_sequence)

    default_probs = np.zeros(len(action_classes))
    no_action_idx = action_classes.index("No Action") if "No Action" in action_classes else 0
    default_probs[no_action_idx] = 1.0

    if normalized_keypoints is None:
        return action_classes[no_action_idx], default_probs

    # Shape check
    if normalized_keypoints.shape != (sequence_length, input_size):
        return action_classes[no_action_idx], default_probs

    # Add batch dimension and send to device
    normalized_keypoints_tensor = torch.tensor(normalized_keypoints, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model returns (probabilities, attention_weights)
        probabilities_tensor, _ = model(normalized_keypoints_tensor)

        # Get probabilities for the single batch item
        probabilities = probabilities_tensor.cpu().numpy()[0]  # Shape (num_classes,)

        predicted_index = np.argmax(probabilities)
        predicted_action = action_classes[predicted_index]

    return predicted_action, probabilities

# ------------------------------------------------------------------------------
# 5. STREAMLIT UI SETUP
# ------------------------------------------------------------------------------
def colored_fall_progress_bar(current, total):
    pct = min(1.0, current / total)
    percent = int(pct * 100)
    # Color logic: green <50%, yellow <80%, red >=80%
    if pct < 0.5:
        color = "#4caf50"  # Green
    elif pct < 0.8:
        color = "#ffc107"  # Amber
    else:
        color = "#f44336"  # Red
    bar_html = f"""
    <div style="margin-top:8px;margin-bottom:8px;">
      <div style="background-color:#eee; border-radius:16px; height:36px; width:100%; box-shadow: 0 2px 8px #0001;">
        <div style="
            width:{percent}%;
            background:{color};
            height:36px;
            border-radius:16px;
            transition: width 0.3s;
            display:flex;
            align-items:center;
            justify-content:center;
            font-weight:bold;
            color:#222;
            font-size:1.1em;">
          {current}/{total} frames ({percent}%)
        </div>
      </div>
    </div>
    """
    return bar_html

def setup_ui():
    """Setup all UI components and return user selections"""
    # App title and description
    st.title("Guardian(AI)👁️")
    st.markdown("Real-time detection and analysis of human actions using advanced computer vision")
    
    # Sidebar configuration
    st.sidebar.header("📊 Controls")
    
    # Load model paths
    pose_model_path = os.environ.get("POSE_MODEL_PATH", "./yolo11n-pose.pt")
    action_model_path = os.environ.get("ACTION_MODEL_PATH", "./models/Guardian_best_model.pth")
    
    # Upload custom model files
    with st.sidebar.expander("Model Settings", expanded=False):
        st.write("Upload custom model files (optional)")
        uploaded_pose_model = st.file_uploader("YOLO Pose Model", type=['pt'])
        uploaded_action_model = st.file_uploader("Action Recognition Model", type=['pth'])
        
        if uploaded_pose_model:
            # Save the uploaded pose model to a temporary file
            temp_pose_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
            temp_pose_model.write(uploaded_pose_model.getvalue())
            pose_model_path = temp_pose_model.name
            
        if uploaded_action_model:
            # Save the uploaded action model to a temporary file
            temp_action_model = tempfile.NamedTemporaryFile(delete=False, suffix='.pth')
            temp_action_model.write(uploaded_action_model.getvalue())
            action_model_path = temp_action_model.name
    
    # Start/Stop buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_button = st.button("▶️ Start")
    with col2:
        stop_button = st.button("⏹️ Stop")
    
    # Camera source selection
    st.sidebar.header("📹 Camera Source")
    input_type = st.sidebar.radio("Input Type", ["Webcam", "IP Camera", "Video File"])
    
    camera_source = 0  # Default webcam
    if input_type == "IP Camera":
        camera_url = st.sidebar.text_input("Camera URL", "http://172.19.102.161:8080/video")
        camera_source = camera_url
    elif input_type == "Video File":
        video_file = st.sidebar.file_uploader("Upload Video", type=['mp4', 'avi', 'mov'])
        if video_file:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            camera_source = tfile.name
    else:  # Webcam
        camera_index = st.sidebar.number_input("Camera Index", min_value=0, max_value=10, value=0, step=1)
        camera_source = int(camera_index)
    
    # Display mode options
    st.sidebar.header("🖥️ Display Mode")
    view_type = st.sidebar.radio("View Type", ["Live Camera"])

    # Alert settings
    st.sidebar.header("⚠️ Alert Settings")
    enable_sound_alerts = st.sidebar.checkbox("Enable Sound Alerts", value=True)
    notify_authorities = st.sidebar.checkbox("Notify Authorities on Falls", value=False)
    save_alert_images = st.sidebar.checkbox("Save Alert Images", value=True)
    
    # Fall detection settings
    with st.sidebar.expander("🚨 Fall Detection Settings", expanded=True):
        st.markdown(
            """
            <div style="background-color:#fff3cd; padding:10px; border-radius:8px; border-left:6px solid #ff9800;">
                <b>Configure how sensitive the system is to falls.</b><br>
                <span style="font-size: 0.95em; color:#856404;">
                    Increase the threshold for fewer false alarms, or decrease for faster alerts.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        fall_threshold = st.slider(
            "🕒 Fall Detection Threshold (frames)",
            min_value=10, max_value=50, value=25, step=5,
            help="Number of consecutive frames of falling action before triggering an alert"
        )


    # Update threshold in session state
    if 'fall_alert_threshold' in st.session_state:
        st.session_state.fall_alert_threshold = fall_threshold
    
    return (start_button, stop_button, camera_source, view_type, 
            enable_sound_alerts, notify_authorities, save_alert_images,
            pose_model_path, action_model_path)

def create_ui_containers():
    """Create and return containers for different parts of the UI"""
    # Main display area with tabs
    tab1, tab2 = st.tabs(["📺 Live View", "🔔 Alert Monitor"])
    
    with tab1:
        # Live view tab
        video_container = st.container()
        metrics_container = st.container()
        
    with tab2:
        # Alert monitor tab
        st.subheader("Alert History")
        alert_container = st.container()
    
    # Log console at the bottom (collapsible)
    with st.expander("System Log", expanded=False):
        status_container = st.container()
        
    return tab1, tab2, video_container, metrics_container, alert_container, status_container

def initialize_session_state():
    """Initialize all session state variables"""
    # Initialize session state if not already done
    if 'run' not in st.session_state:
        st.session_state.run = False
        
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
        
    if 'keypoint_buffer' not in st.session_state:
        st.session_state.keypoint_buffer = deque(maxlen=35)  # frames_per_clip
        
    if 'predicted_action' not in st.session_state:
        st.session_state.predicted_action = "Initializing..."
        
    if 'action_duration' not in st.session_state:
        st.session_state.action_duration = 0
        
    if 'action_start_time' not in st.session_state:
        st.session_state.action_start_time = None
        
    if 'previous_action' not in st.session_state:
        st.session_state.previous_action = None
        
    if 'last_probabilities' not in st.session_state:
        st.session_state.last_probabilities = None
        
    if 'plot_data' not in st.session_state:
        st.session_state.plot_data = {
            'frame_numbers': deque(maxlen=35),
            'probabilities': []
        }
        
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
        
    # New variables for consecutive fall detection
    if 'consecutive_fall_frames' not in st.session_state:
        st.session_state.consecutive_fall_frames = 0
        
    if 'fall_alert_threshold' not in st.session_state:
        st.session_state.fall_alert_threshold = 50  # 50 consecutive frames to trigger alert
        
    if 'fall_alert_triggered' not in st.session_state:
        st.session_state.fall_alert_triggered = False
        
    if 'total_falls_detected' not in st.session_state:
        st.session_state.total_falls_detected = 0

# def create_placeholders(video_container, metrics_container, action_classes):
#     """Create all UI placeholders for updating during processing"""
#     with video_container:
#         video_placeholder = st.empty()
        
#     with metrics_container:
#         col1, col2 = st.columns(2)
#         with col1:
#             action_text = st.empty()
#             duration_text = st.empty()
            
#             # Add fall detection counter display
#             fall_counter_container = st.container()
#             with fall_counter_container:
#                 fall_counter_text = st.empty()
#                 fall_counter_progress = st.empty()
#                 total_falls_text = st.empty()
                
#             fps_text = st.empty()
#             probability_bars = []
#             for _ in action_classes:
#                 probability_bars.append(st.empty())
#         with col2:
#             plot_placeholder = st.empty()
        
#     return (video_placeholder, action_text, duration_text, fps_text, probability_bars, 
#             plot_placeholder, fall_counter_text, fall_counter_progress, total_falls_text)
def create_placeholders(video_container, metrics_container, action_classes):
    """Create all UI placeholders for updating during processing"""
    with video_container:
        col1, col2 = st.columns([1, 1])  # Video | Metrics
        with col1:
            video_placeholder = st.empty()
        with col2:
            action_text = st.empty()
            duration_text = st.empty()
            fall_counter_text = st.empty()
            fall_counter_progress = st.empty()
            total_falls_text = st.empty()
            fps_text = st.empty()
            probability_bars = []
            for _ in action_classes:
                probability_bars.append(st.empty())
            # Place the plot directly below the probability bars
            plot_placeholder = st.empty()
    # metrics_container is not used anymore for the plot
    return (video_placeholder, action_text, duration_text, fps_text, probability_bars, 
            plot_placeholder, fall_counter_text, fall_counter_progress, total_falls_text)
# ------------------------------------------------------------------------------
# 6. PROCESSING FUNCTIONS
# ------------------------------------------------------------------------------

def update_alert_monitor(alert_data, alert_container, tab2):
    """Update the alert monitor tab with new alert data"""
    with tab2:
        with alert_container:
            # Clear and re-render the alerts
            alert_container.empty()
            for alert in st.session_state.alerts:
                alert_container.markdown(f"**{alert['type']} #{alert.get('fall_number', '?')}** - {alert['timestamp']}")
                alert_container.markdown(f"Action: {alert['action']} (Confidence: {alert['confidence']:.2f})")
                if 'duration' in alert:
                    alert_container.markdown(f"Duration: {alert['duration']}")
                if 'consecutive_frames' in alert:
                    alert_container.markdown(f"Consecutive frames: {alert['consecutive_frames']}")
                alert_container.markdown("---")

def handle_fall_detection(predicted_action, current_probabilities, annotated_frame, 
                         enable_sound_alerts, notify_authorities, save_alert_images,
                         alert_container, tab2):
    """Handle fall detection, alerts, and notifications"""
    # Update consecutive fall frames counter
    if predicted_action == "Falling":
        # Increment the consecutive fall frames counter
        st.session_state.consecutive_fall_frames += 1
        
        # Only trigger alert if threshold is reached and no alert has been triggered for this sequence
        if (st.session_state.consecutive_fall_frames >= st.session_state.fall_alert_threshold and 
            not st.session_state.fall_alert_triggered and enable_sound_alerts):
            
            # Set triggered flag to prevent multiple alerts for same fall
            st.session_state.fall_alert_triggered = True
            
            # Increment total falls detected counter
            st.session_state.total_falls_detected += 1
            
            # Use toast notification
            st.toast(f"⚠️ FALL DETECTED! (Fall #{st.session_state.total_falls_detected})", icon="⚠️")
            
            # Capture timestamp for the alert
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Create alert data
            alert_data = {
                "type": "Fall Detection",
                "timestamp": timestamp,
                "action": predicted_action,
                "confidence": float(current_probabilities[0]),
                "duration": f"{st.session_state.action_duration:.1f} seconds",
                "consecutive_frames": st.session_state.consecutive_fall_frames,
                "fall_number": st.session_state.total_falls_detected
            }
            
            # Add to alerts history
            st.session_state.alerts.append(alert_data)
            
            # Update alert monitor tab
            update_alert_monitor(alert_data, alert_container, tab2)
            
            # Alert logic for authorities notification
            if notify_authorities:
                st.toast("🚨 Notifying authorities...", icon="🚨")
            
            # Save alert image if enabled
            if save_alert_images:
                img_timestamp = time.strftime("%Y%m%d-%H%M%S")
                # In a real app you'd save to disk, here we just log the action
                st.toast(f"Fall #{st.session_state.total_falls_detected} image saved: fall_alert_{img_timestamp}.jpg", icon="💾")
    else:
        # Reset counter and triggered flag if not falling
        st.session_state.consecutive_fall_frames = 0
        st.session_state.fall_alert_triggered = False

def update_action_duration(predicted_action):
    """Update action duration tracking"""
    if st.session_state.previous_action != predicted_action:
        # Action changed, reset timer
        st.session_state.action_start_time = time.time()
        st.session_state.action_duration = 0
    else:
        # Same action continuing, update duration
        if st.session_state.action_start_time is not None:
            st.session_state.action_duration = time.time() - st.session_state.action_start_time
    
    # Update action state
    st.session_state.previous_action = predicted_action

def update_plots_and_ui(fig, ax, lines, plot_placeholder, action_text, duration_text, 
                        fps_text, probability_bars, action_classes, loop_start,
                        fall_counter_text, fall_counter_progress, total_falls_text):
    """Update probability plots and UI elements"""
    # Update action prediction text, duration and FPS
    action_text.markdown(f"### Action: **{st.session_state.predicted_action}**")
    duration_text.markdown(f"**Duration**: {st.session_state.action_duration:.1f} seconds")
    
    # Update fall counter display
    pct = st.session_state.consecutive_fall_frames / st.session_state.fall_alert_threshold if st.session_state.fall_alert_threshold > 0 else 0

    # --- Sound zone logic ---
    if 'last_sound_zone' not in st.session_state:
        st.session_state.last_sound_zone = None

    if st.session_state.predicted_action == "Falling":
        # Determine zone
        sound_zone = None
        if 0.5 <= pct < 0.8:
            sound_zone = "yellow"
        elif pct >= 0.8:
            sound_zone = "red"
        else:
            sound_zone = None

        # Play sound only when entering a new zone
        if sound_zone != st.session_state.last_sound_zone:
            if sound_zone == "yellow":
                pygame.mixer.music.load("sounds\quotwarningquot-175692.mp3")
                pygame.mixer.music.play()
            elif sound_zone == "red":
                pygame.mixer.music.load("sounds/australia-eas-alarm-267664.mp3")
                pygame.mixer.music.play()
            st.session_state.last_sound_zone = sound_zone
        elif sound_zone is None:
            st.session_state.last_sound_zone = None

        fall_counter_text.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: -8px;">
                <span style="font-size:1.2em; margin-right: 6px;">🚨</span>
                <span style="font-size:1em; font-weight:600;">{st.session_state.consecutive_fall_frames}/{st.session_state.fall_alert_threshold} frames ({int(100*st.session_state.consecutive_fall_frames/st.session_state.fall_alert_threshold)}%)</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        progress_html = colored_fall_progress_bar(
            st.session_state.consecutive_fall_frames,
            st.session_state.fall_alert_threshold
        )
        fall_counter_progress.markdown(progress_html, unsafe_allow_html=True)
    else:
        st.session_state.last_sound_zone = None
        fall_counter_text.markdown(
            '<div style="display: flex; justify-content: flex-end; align-items: center; margin-bottom: -8px;">'
            '<span style="font-size:1.2em; margin-right: 6px; color: #bbb;">🚨</span>'
            '<span style="font-size:1em; font-weight:600; color: #888;">Not falling</span>'
            '</div>',
            unsafe_allow_html=True
        )
        progress_html = colored_fall_progress_bar(0, st.session_state.fall_alert_threshold)
        fall_counter_progress.markdown(progress_html, unsafe_allow_html=True)
    
    # Display total falls counter
    total_falls_text.markdown(f"**Total Falls Detected**: {st.session_state.total_falls_detected}")
    
    # Calculate FPS
    loop_time = time.time() - loop_start
    current_fps = 1.0 / loop_time if loop_time > 0 else 0
    fps_text.text(f"FPS: {current_fps:.1f}")
    
    # Display probability bars
    for i, (bar, action) in enumerate(zip(probability_bars, action_classes)):
        prob = st.session_state.last_probabilities[i]
        bar.progress(float(prob), text=f"{action}: {prob:.2f}")
    
    # Update probability plot
    if len(st.session_state.plot_data['frame_numbers']) > 0:
        try:
            for i, line in enumerate(lines):
                line.set_data(
                    list(st.session_state.plot_data['frame_numbers']),
                    list(st.session_state.plot_data['probabilities'][i])
                )
            if st.session_state.plot_data['frame_numbers']:
                ax.set_xlim(
                    st.session_state.plot_data['frame_numbers'][0],
                    st.session_state.plot_data['frame_numbers'][-1] + 1
                )
            ax.relim()
            ax.autoscale_view(scalex=False, scaley=True)
            plot_img = plot_to_image(fig)
            plot_placeholder.image(plot_img, use_container_width=False)
        except Exception as e:
            st.error(f"Error updating plot: {e}")
# ------------------------------------------------------------------------------
# 7. MAIN PROCESSING LOOP
# ------------------------------------------------------------------------------

def process_video(cap, pose_model, action_model, device, action_classes, frames_per_clip, input_size,
                 video_placeholder, action_text, duration_text, fps_text, probability_bars, 
                 plot_placeholder, alert_container, status_container,
                 fig, ax, lines, tab2,
                 enable_sound_alerts, notify_authorities, save_alert_images,
                 fall_counter_text, fall_counter_progress, total_falls_text):
    """Main video processing loop"""
    while st.session_state.run:
        loop_start = time.time()
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            with status_container:
                st.error("Can't receive frame from queue. Exiting...")
            break

        
        # Resize frame to a common size
        frame = cv2.resize(frame, (640, 480))
        
        # Pose Estimation
        results = pose_model(frame, verbose=False, conf=0.5)
        
        # Annotate frame with pose
        if len(results) > 0:
            annotated_frame = results[0].plot()
        else:
            annotated_frame = frame
        
        # Extract keypoints
        current_frame_keypoints = []
        try:
            if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
                current_frame_keypoints = results[0].keypoints.data.cpu().numpy().tolist()
                # If no keypoints detected, treat as 'No Action'
                if not current_frame_keypoints:
                    st.session_state.predicted_action = "No Action"
                    st.session_state.last_probabilities = np.array([0.0, 1.0, 0.0])  # Assuming order: ["Falling", "No Action", "Waving"]
                    update_action_duration("No Action")
                    continue  # Skip the rest of the loop for this frame

                if len(current_frame_keypoints) > 0 and not isinstance(current_frame_keypoints[0], list):
                    current_frame_keypoints = [current_frame_keypoints]
        except Exception as e:
            with status_container:
                st.error(f"Error extracting keypoints: {e}")
            current_frame_keypoints = []
        
        # Add to keypoint buffer
        # Only append keypoints if at least one person is detected
        if current_frame_keypoints:
            st.session_state.keypoint_buffer.append({
                "frame_num": st.session_state.frame_count,
                "keypoints": current_frame_keypoints
            })
        else:
            # No people detected — reset buffer and prediction
            st.session_state.keypoint_buffer.clear()
            st.session_state.predicted_action = "No Action"
            st.session_state.last_probabilities = np.array([0.0, 1.0, 0.0])  # Adjust if your class order is different
            update_action_duration("No Action")
        
        # Force UI update to reflect "No Action"
        update_plots_and_ui(
            fig, ax, lines, plot_placeholder, action_text, 
            duration_text, fps_text, probability_bars, action_classes, loop_start,
            fall_counter_text, fall_counter_progress, total_falls_text
        )

        # Action Prediction (only if buffer is full)
        if len(st.session_state.keypoint_buffer) == frames_per_clip:
            keypoints_for_prediction = list(st.session_state.keypoint_buffer)
            predicted_action, current_probabilities = predict_action(
                action_model,
                keypoints_for_prediction,
                action_classes,
                device,
                frames_per_clip,
                input_size
            )
            
            # Update prediction data and duration
            st.session_state.predicted_action = predicted_action
            st.session_state.last_probabilities = current_probabilities
            update_action_duration(predicted_action)
            
            # Update plot data
            st.session_state.plot_data['frame_numbers'].append(st.session_state.frame_count)
            for i, prob in enumerate(current_probabilities):
                st.session_state.plot_data['probabilities'][i].append(prob)
            
            # Handle fall detection based on predicted action
            handle_fall_detection(
                predicted_action, current_probabilities, annotated_frame,
                enable_sound_alerts, notify_authorities, save_alert_images,
                alert_container, tab2
            )
        
        # Update probability plots and UI elements
        update_plots_and_ui(
            fig, ax, lines, plot_placeholder, action_text, 
            duration_text, fps_text, probability_bars, action_classes, loop_start,
            fall_counter_text, fall_counter_progress, total_falls_text
        )
        
        # Display video frame
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        rgb_frame_resized = cv2.resize(rgb_frame, (512, 350))
        video_placeholder.image(rgb_frame_resized, use_container_width=True)
        
        # Increment frame counter
        st.session_state.frame_count += 1
        
        # Check if stop button pressed (via session state)
        if not st.session_state.run:
            break

def load_models(pose_model_path, action_model_path, status_container):
    """Load and initialize the models"""
    with status_container:
        try:
            # Device setup
            st.info(f"Using {device} for model inference")
            
            # Load YOLO pose model
            with st.spinner("Loading YOLOv8 pose detection model..."):
                pose_model = YOLO(pose_model_path)
                st.success("✅ YOLOv8 pose model loaded successfully")
            
            # Model configuration
            action_classes = ["Falling", "No Action", "Waving"]
            frames_per_clip = 35
            input_size = 34  # 17 keypoints * 2 (x and y)
            hidden_size = 256
            num_layers = 4
            num_classes = len(action_classes)
            dropout_rate = 0.5
            
            # Initialize BiLSTM model
            with st.spinner("Initializing action recognition model..."):
                action_model = ActionRecognitionBiLSTMWithAttention(
                    input_size, hidden_size, num_layers, num_classes, dropout_rate
                ).to(device)
                
                # Load action model weights
                st.info(f"Loading model weights from {action_model_path}...")
                action_model.load_state_dict(torch.load(action_model_path, map_location=device))
                action_model.eval()
                st.success("✅ Action recognition model weights loaded successfully")
            
            # Initialize session state for probabilities if not already done
            if st.session_state.last_probabilities is None:
                st.session_state.last_probabilities = np.zeros(len(action_classes))
                st.session_state.plot_data['probabilities'] = [deque(maxlen=frames_per_clip) for _ in action_classes]
            
            st.success("✅ Models initialized successfully. Ready for action recognition!")
            
            return pose_model, action_model, device, action_classes, frames_per_clip, input_size
            
        except Exception as e:
            st.error(f"Error loading models: {e}")
            st.session_state.run = False
            raise e

# ------------------------------------------------------------------------------
# 8. MAIN FUNCTION
# ------------------------------------------------------------------------------

def main():
    """Main application function"""
    # Initialize session state
    initialize_session_state()
    
    # Setup UI components and get user settings
    (start_button, stop_button, camera_source, view_type, 
     enable_sound_alerts, notify_authorities, save_alert_images,
     pose_model_path, action_model_path) = setup_ui()
    
    # Create UI containers
    tab1, tab2, video_container, metrics_container, alert_container, status_container = create_ui_containers()
    
    # Handle start/stop logic
    if start_button:
        st.session_state.run = True
    if stop_button:
        st.session_state.run = False
    
    # Main processing logic
    if st.session_state.run:
        try:
            # Load models
            pose_model, action_model, device, action_classes, frames_per_clip, input_size = load_models(
                pose_model_path, action_model_path, status_container
            )
            
            # Setup video capture
            cap = cv2.VideoCapture(camera_source)
            if not cap.isOpened():
                st.error(f"Error: Cannot open video source {camera_source}")
                st.session_state.run = False
                return

            # Start the frame grabber thread
            grabber_thread = threading.Thread(target=frame_grabber, args=(cap, frame_queue, stop_event), daemon=True)
            grabber_thread.start()
                        
            # Setup visualization
            fig, ax = plt.subplots(figsize=(4, 2.5))
            lines = []
            colors = plt.cm.viridis(np.linspace(0, 1, len(action_classes)))
            for i, action in enumerate(action_classes):
                line, = ax.plot([], [], label=action, color=colors[i], linewidth=2)
                lines.append(line)
            ax.set_xlabel('Frame Number')
            ax.set_ylabel('Probability')
            ax.set_title('Action Probabilities')
            ax.set_ylim([0, 1.1])
            ax.legend(loc='upper left')
            ax.grid(True)
            fig.tight_layout()
            
            # Create UI placeholders
            (video_placeholder, action_text, duration_text, fps_text, 
             probability_bars, plot_placeholder,
             fall_counter_text, fall_counter_progress, total_falls_text) = create_placeholders(
                video_container, metrics_container, action_classes
            )
            
            # Start processing loop
            process_video(
                cap, pose_model, action_model, device, action_classes, frames_per_clip, input_size,
                video_placeholder, action_text, duration_text, fps_text, probability_bars, 
                plot_placeholder, alert_container, status_container,
                fig, ax, lines, tab2,
                enable_sound_alerts, notify_authorities, save_alert_images,
                fall_counter_text, fall_counter_progress, total_falls_text
            )
            
            # Cleanup
            cap.release()
            
        except Exception as e:
            with status_container:
                st.error(f"Error: {str(e)}")
            st.session_state.run = False
    else:
        # Not running - display start message
        with tab1:
            st.info("Click the Start button to begin action recognition.")

if __name__ == "__main__":
    main()