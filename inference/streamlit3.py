import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import time
from collections import deque, defaultdict
import threading
import queue
import pygame
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import base64

# Initialize pygame mixer for sound alerts
pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)

# Set device for model inference
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Email thread pool
email_executor = ThreadPoolExecutor(max_workers=3)

# Constants
DEFAULT_RECIPIENT_EMAIL = "bahri1140@hotmail.com"
DEFAULT_SMTP_SERVER = "smtp.gmail.com"
DEFAULT_SMTP_PORT = "587"

# Performance settings
INFERENCE_INTERVAL = 5  # Run YOLO every N frames
DISPLAY_QUEUE_SIZE = 2  # Keep display queue small for low latency
CAPTURE_QUEUE_SIZE = 5  # Small buffer for captured frames

# Create alert sound
def create_alert_sound():
    """Create an alert sound using numpy arrays"""
    sample_rate = 22050
    duration = 0.5  # seconds
    frequency = 800  # Hz
    
    frames = int(duration * sample_rate)
    arr = np.zeros((frames, 2), dtype=np.int16)
    max_sample = 2**(16 - 1) - 1
    
    for i in range(frames):
        sample = int(max_sample * np.sin(2 * np.pi * frequency * i / sample_rate))
        arr[i] = [sample, sample]
    
    return pygame.sndarray.make_sound(arr)

# Initialize alert sounds
try:
    # Load warning sound (for half bar)
    warning_sound_path = "inference/sounds/warning.mp3"
    if os.path.exists(warning_sound_path):
        warning_sound = pygame.mixer.Sound(warning_sound_path)
        print(f"✅ Successfully loaded warning sound: {warning_sound_path}")
    else:
        warning_sound = create_alert_sound()
        print("⚠️ Warning sound file not found, using generated sound")
    
    # Load alarm sound (for full bar)
    alarm_sound_path = "inference/sounds/alarm.mp3"
    if os.path.exists(alarm_sound_path):
        alarm_sound = pygame.mixer.Sound(alarm_sound_path)
        print(f"✅ Successfully loaded alarm sound: {alarm_sound_path}")
    else:
        alarm_sound = create_alert_sound()
        print("⚠️ Alarm sound file not found, using generated sound")
        
except (pygame.error, FileNotFoundError) as e:
    print(f"❌ Could not load sound files: {e}")
    # Fallback to generated sounds
    warning_sound = create_alert_sound()
    alarm_sound = create_alert_sound()
    print("✅ Using generated sounds as fallback")

def test_sounds():
    """Test both alert sounds"""
    try:
        print("🔊 Testing warning sound...")
        warning_sound.play()
        time.sleep(1)
        print("🔊 Testing alarm sound...")
        alarm_sound.play()
        print("✅ Sound tests successful!")
        return True
    except Exception as e:
        print(f"❌ Sound test failed: {e}")
        return False

# ------------------------------------------------------------------------------
# MODEL ARCHITECTURE (from your existing code)
# ------------------------------------------------------------------------------

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1)

    def forward(self, lstm_output):
        scores = torch.tanh(self.attention_weights(lstm_output))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights.squeeze(-1)

class ActionRecognitionBiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(ActionRecognitionBiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        lstm_out = self.dropout(lstm_out)
        context_vector, attention_weights = self.attention(lstm_out)
        context_vector_dropped = self.dropout(context_vector)
        logits = self.fc(context_vector_dropped)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities, attention_weights

# ------------------------------------------------------------------------------
# POSE PREPROCESSOR (from great_multi.py)
# ------------------------------------------------------------------------------

class PosePreprocessor:
    def __init__(self, max_frames=35, feature_size=34):
        self.max_frames = max_frames
        self.feature_size = feature_size
        self.conf_threshold = 0.2
        self.alpha = 0.8

    def process_keypoints_sequence(self, keypoints_sequence):
        all_frames_keypoints = []
        previous_frame_normalized = None
        
        relevant_keypoints_data = keypoints_sequence[-self.max_frames:]
        
        for frame_data in relevant_keypoints_data:
            processed_frame = np.zeros(self.feature_size)
            try:
                if 'keypoints_for_id' in frame_data:
                    frame_keypoints_np = np.array(frame_data['keypoints_for_id']).reshape(-1, 3)
                    if frame_keypoints_np.shape != (17, 3):
                        all_frames_keypoints.append(processed_frame)
                        continue
                    
                    valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > self.conf_threshold]
                    if valid_keypoints.shape[0] < 2:
                        all_frames_keypoints.append(processed_frame)
                        continue
                    
                    mean_x = np.mean(valid_keypoints[:, 0])
                    std_x = np.std(valid_keypoints[:, 0]) + 1e-8
                    mean_y = np.mean(valid_keypoints[:, 1])
                    std_y = np.std(valid_keypoints[:, 1]) + 1e-8
                    
                    normalized_frame_keypoints = frame_keypoints_np.copy()
                    normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
                    normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y
                    
                    if previous_frame_normalized is not None:
                        normalized_frame_keypoints[:, :2] = self.alpha * normalized_frame_keypoints[:, :2] + (1 - self.alpha) * previous_frame_normalized[:, :2]
                    
                    previous_frame_normalized = normalized_frame_keypoints
                    processed_frame = normalized_frame_keypoints[:, :2].flatten()
                    
                    if processed_frame.shape[0] != self.feature_size:
                        processed_frame = np.zeros(self.feature_size)
            except Exception:
                processed_frame = np.zeros(self.feature_size)
                
            all_frames_keypoints.append(processed_frame)
        
        num_processed = len(all_frames_keypoints)
        if num_processed == 0:
            return None
            
        padded_keypoints = np.zeros((self.max_frames, self.feature_size))
        start_idx_pad = max(0, self.max_frames - num_processed)
        
        for i in range(num_processed):
            if all_frames_keypoints[i].shape[0] == self.feature_size:
                padded_keypoints[start_idx_pad + i, :] = all_frames_keypoints[i]
            else:
                padded_keypoints[start_idx_pad + i, :] = np.zeros(self.feature_size)
                
        return padded_keypoints

# ------------------------------------------------------------------------------
# HIGH PERFORMANCE VIDEO PROCESSOR
# ------------------------------------------------------------------------------

class HighPerformanceVideoProcessor:
    def __init__(self, pose_model, action_model, action_classes, device):
        self.pose_model = pose_model
        self.action_model = action_model
        self.action_classes = action_classes
        self.device = device
        self.frames_buffer_size = 35
        self.input_size = 34

        
        # Tracking data
        self.tracked_persons_data = defaultdict(lambda: {
            'keypoint_buffer': deque(maxlen=self.frames_buffer_size),
            'action': "Initializing...",
            'probabilities': np.zeros(len(self.action_classes)),
            'last_seen': time.time(),
            'color': self.random_non_red_color(),
            'bbox': None,
            'consecutive_fall_frames': 0,
            'fall_alert_triggered': False,
            'fall_start_time': None,
            'last_keypoints': None,
            'last_track_id': None,
            'warning_sound_played': False,
            'alarm_sound_played': False
        })
        
        self.preprocessor = PosePreprocessor(max_frames=self.frames_buffer_size, feature_size=self.input_size)
        self.frame_count = 0
        self.tracked_persons_lock = threading.Lock()
        
        # Performance optimization
        self.last_detection_result = None
        self.last_annotated_frame = None
        self.active_count = 0
        
        # Threading
        self.capture_queue = queue.Queue(maxsize=CAPTURE_QUEUE_SIZE)
        self.display_queue = queue.Queue(maxsize=DISPLAY_QUEUE_SIZE)
        self.processing_thread = None
        self.capture_thread = None
        self.stop_threads = False

    def random_non_red_color(self):
        while True:
            color = (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200))
            if color != (0, 0, 255):
                return color
        
        
    def predict_person_action(self, person_keypoints_sequence):
        self.action_model.eval()
        normalized_features = self.preprocessor.process_keypoints_sequence(person_keypoints_sequence)
        default_probs = np.zeros(len(self.action_classes))
        no_action_idx = self.action_classes.index("No Action") if "No Action" in self.action_classes else 0
        default_probs[no_action_idx] = 1.0
        default_action = self.action_classes[no_action_idx]
        
        if normalized_features is None:
            return default_action, default_probs
            
        input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities_tensor, _ = self.action_model(input_tensor)
            probabilities = probabilities_tensor.cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        predicted_action = self.action_classes[predicted_index]
        return predicted_action, probabilities
    
    def capture_frames(self, cap):
        """High-speed frame capture thread"""
        while not self.stop_threads:
            ret, frame = cap.read()
            if ret:
                # Drop old frames to maintain real-time performance
                if self.capture_queue.full():
                    try:
                        self.capture_queue.get_nowait()
                    except:
                        pass
                self.capture_queue.put(frame)
    
    def process_frames(self):
        """Frame processing thread with optimized inference"""
        fall_detections = []
        
        while not self.stop_threads:
            try:
                frame = self.capture_queue.get(timeout=0.1)
                self.frame_count += 1
                
                # Run YOLO detection only at intervals
                if self.frame_count % INFERENCE_INTERVAL == 0:
                    # Run detection
                    results = self.pose_model.track(
                        frame,
                        persist=True,
                        conf=0.25,
                        iou=0.20,
                        tracker="inference/bytetrack.yaml",
                        verbose=False,
                        imgsz=640  # Smaller size for speed
                    )
                    self.last_detection_result = results
                
                # Process and annotate frame
                annotated_frame, new_fall_detections = self.process_single_frame(frame)
                fall_detections.extend(new_fall_detections)
                
                # Send to display queue
                if self.display_queue.full():
                    try:
                        self.display_queue.get_nowait()
                    except:
                        pass
                
                self.display_queue.put((annotated_frame, fall_detections))
                fall_detections = []  # Clear after sending
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Processing error: {e}")
    
    def process_single_frame(self, frame):
        """Process a single frame with current detection results"""
        current_time = time.time()
        fall_detections = []
        current_frame_ids = []
        
        # Use last detection result if available
        if self.last_detection_result is not None:
            results = self.last_detection_result
            
            persons_detected = (results[0].boxes is not None and 
                              results[0].boxes.id is not None and 
                              results[0].keypoints is not None and 
                              len(results[0].boxes) > 0)
            
            if persons_detected:
                boxes = results[0].boxes.data.cpu().numpy()
                keypoints = results[0].keypoints.data.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                min_len = min(len(boxes), len(keypoints), len(track_ids))
                
                for i in range(min_len):
                    person_id = track_ids[i]
                    current_frame_ids.append(person_id)
                    person_keypoints = keypoints[i]
                    person_bbox = boxes[i][:4]
                    
                    with self.tracked_persons_lock:
                        self.tracked_persons_data[person_id]['last_seen'] = current_time
                        self.tracked_persons_data[person_id]['bbox'] = person_bbox
                        self.tracked_persons_data[person_id]['last_keypoints'] = person_keypoints
                        self.tracked_persons_data[person_id]['last_track_id'] = person_id
                        
                        # Only update keypoint buffer on inference frames
                        if self.frame_count % INFERENCE_INTERVAL == 0:
                            frame_data_for_buffer = {
                                "frame_num": self.frame_count,
                                "keypoints_for_id": person_keypoints.tolist()
                            }
                            self.tracked_persons_data[person_id]['keypoint_buffer'].append(frame_data_for_buffer)
                            
                            if len(self.tracked_persons_data[person_id]['keypoint_buffer']) == self.frames_buffer_size:
                                buffer_to_predict = list(self.tracked_persons_data[person_id]['keypoint_buffer'])
                                try:
                                    pred_action, pred_probs = self.predict_person_action(buffer_to_predict)
                                    self.tracked_persons_data[person_id]['action'] = pred_action
                                    self.tracked_persons_data[person_id]['probabilities'] = pred_probs
                                    
                                    # Handle fall detection
                                    if pred_action == "Falling":
                                        self.tracked_persons_data[person_id]['consecutive_fall_frames'] += 1
                                        if self.tracked_persons_data[person_id]['fall_start_time'] is None:
                                            self.tracked_persons_data[person_id]['fall_start_time'] = time.time()
                                        
                                        # Play warning sound at half bar (12 frames)
                                        if (self.tracked_persons_data[person_id]['consecutive_fall_frames'] == 12 and 
                                            not self.tracked_persons_data[person_id]['warning_sound_played']):
                                            self.tracked_persons_data[person_id]['warning_sound_played'] = True
                                            
                                            # Capture warning frame image
                                            x1, y1, x2, y2 = map(int, person_bbox)
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                                            warning_frame = frame[y1:y2, x1:x2].copy()
                                            
                                            fall_detections.append({
                                                'person_id': person_id,
                                                'bbox': person_bbox,
                                                'confidence': pred_probs[0],
                                                'duration': time.time() - self.tracked_persons_data[person_id]['fall_start_time'],
                                                'consecutive_frames': self.tracked_persons_data[person_id]['consecutive_fall_frames'],
                                                'frame_image': warning_frame,
                                                'alert_type': 'warning'
                                            })
                                        
                                        # Trigger final alarm at full bar (25 frames)
                                        if (self.tracked_persons_data[person_id]['consecutive_fall_frames'] >= 25 and 
                                            not self.tracked_persons_data[person_id]['fall_alert_triggered']):
                                            self.tracked_persons_data[person_id]['fall_alert_triggered'] = True
                                            self.tracked_persons_data[person_id]['alarm_sound_played'] = True
                                            fall_duration = time.time() - self.tracked_persons_data[person_id]['fall_start_time']
                                            
                                            # Capture fall frame image
                                            x1, y1, x2, y2 = map(int, person_bbox)
                                            x1, y1 = max(0, x1), max(0, y1)
                                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                                            fall_frame = frame[y1:y2, x1:x2].copy()
                                            
                                            fall_detections.append({
                                                'person_id': person_id,
                                                'bbox': person_bbox,
                                                'confidence': pred_probs[0],
                                                'duration': fall_duration,
                                                'consecutive_frames': self.tracked_persons_data[person_id]['consecutive_fall_frames'],
                                                'frame_image': fall_frame,
                                                'alert_type': 'alarm'
                                            })
                                    else:
                                        # Reset fall detection state
                                        self.tracked_persons_data[person_id]['consecutive_fall_frames'] = 0
                                        self.tracked_persons_data[person_id]['fall_alert_triggered'] = False
                                        self.tracked_persons_data[person_id]['fall_start_time'] = None
                                        self.tracked_persons_data[person_id]['warning_sound_played'] = False
                                        self.tracked_persons_data[person_id]['alarm_sound_played'] = False
                                    
                                except Exception as e:
                                    print(f"Error during action prediction for ID {person_id}: {e}")
        
        # Clean up old tracks
        ids_to_remove = []
        with self.tracked_persons_lock:
            for pid, data in self.tracked_persons_data.items():
                if current_time - data['last_seen'] > 3.0:  # Reduced timeout for responsiveness
                    ids_to_remove.append(pid)
            for pid in ids_to_remove:
                del self.tracked_persons_data[pid]
        
        # Update active count
        self.active_count = len(current_frame_ids)
        
        # Fast annotation - only draw essentials
        annotated_frame = self.draw_fast_annotations(frame, current_frame_ids)
        
        return annotated_frame, fall_detections
    
    def draw_fast_annotations(self, frame, current_frame_ids):
        """Optimized drawing for performance"""
        annotated_frame = frame.copy()
        
        with self.tracked_persons_lock:
            for person_id in current_frame_ids:
                if person_id in self.tracked_persons_data:
                    data = self.tracked_persons_data[person_id]
                    if data['bbox'] is not None:
                        x1, y1, x2, y2 = map(int, data['bbox'])
                        action = data['action']
                        
                        # Simple color coding
                        if action == "Falling":
                            color = (0, 0, 255)  # Red
                        else:
                            color = data['color']  # Cyan
                        
                        # Simple rectangle and label
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Action label with increased font size
                        action_label = f"[{action}]"
                        font_scale = 0.7  # Increased font scale
                        thickness = 2
                        (text_width, text_height), _ = cv2.getTextSize(action_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                        
                        # Background for the action label for better visibility
                        cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1 - 10), (0,0,0), -1) # Black background
                        cv2.putText(annotated_frame, action_label, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                        id_font_scale = 0.7  # Increased font scale

                        # ID label (optional, can be smaller or positioned differently)
                        id_label = f"ID:{person_id}"
                        cv2.putText(annotated_frame, id_label, (x1, y1 - text_height - 20), # Position above action label
                                  cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, color, 1) # Slightly smaller font for ID
                        
                        # Progress bar for falling - RESTORED!
                        if action == "Falling" and data['consecutive_fall_frames'] > 0:
                            bar_width = x2 - x1
                            bar_height = 8
                            bar_y = y2 + 8
                            progress = min(data['consecutive_fall_frames'] / 25.0, 1.0)
                            progress_width = int(bar_width * progress)
                            
                            # Background bar
                            cv2.rectangle(annotated_frame, (x1, bar_y), 
                                        (x2, bar_y + bar_height), (50, 50, 50), -1)
                            
                            # Progress bar - color changes based on progress
                            if progress < 0.5:  # First half - yellow/orange
                                bar_color = (0, 165, 255)  # Orange
                            else:  # Second half - red (danger)
                                bar_color = (0, 0, 255)  # Red
                            
                            if progress_width > 0:
                                cv2.rectangle(annotated_frame, (x1, bar_y), 
                                            (x1 + progress_width, bar_y + bar_height), 
                                            bar_color, -1)
                            
                            # Frame count text
                            frame_text = f"{data['consecutive_fall_frames']}/25"
                            cv2.putText(annotated_frame, frame_text, 
                                      (x1, bar_y + bar_height + 15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        return annotated_frame
    
    def start_processing(self, cap):
        """Start the processing threads"""
        self.stop_threads = False
        self.capture_thread = threading.Thread(target=self.capture_frames, args=(cap,))
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.capture_thread.start()
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop the processing threads"""
        self.stop_threads = True
        if self.capture_thread:
            self.capture_thread.join()
        if self.processing_thread:
            self.processing_thread.join()
    
    def get_display_frame(self, timeout=0.033):  # ~30 FPS
        """Get the latest frame for display"""
        try:
            return self.display_queue.get(timeout=timeout)
        except queue.Empty:
            return None, []

# ------------------------------------------------------------------------------
# EMAIL FUNCTIONS
# ------------------------------------------------------------------------------

def send_email_alert_async(recipient_email, smtp_server, smtp_port, sender_email, sender_password, 
                          person_id, fall_number, timestamp, image_data=None, confidence=0.0, 
                          duration=0, consecutive_frames=0):
    """Send an email alert asynchronously"""
    try:
        message = MIMEMultipart()
        message["Subject"] = f"🚨 GUARDIAN AI: FALL DETECTED - Person #{person_id}"
        message["From"] = sender_email
        message["To"] = recipient_email
        
        body = f"""
        🚨 FALL DETECTION ALERT 🚨
        
        Guardian AI has detected a fall incident.
        
        📊 DETECTION DETAILS:
        • Person ID: {person_id}
        • Fall Event: #{fall_number}
        • Timestamp: {timestamp}
        • Confidence: {confidence:.2%}
        • Duration: {duration:.1f} seconds
        • Consecutive Frames: {consecutive_frames}
        
        ⚡ IMMEDIATE ACTION REQUIRED
        
        ---------------------------------
        Guardian AI Multi-Person Detection
        """
        message.attach(MIMEText(body, "plain"))
        
        if image_data is not None:
            image = MIMEImage(image_data)
            image.add_header('Content-Disposition', 'attachment', 
                           filename=f'fall_person_{person_id}_{timestamp.replace(":", "-")}.jpg')
            message.attach(image)
        
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        return True, "Email sent successfully"
    except Exception as e:
        return False, str(e)

# Function to convert image to base64 for HTML display
def image_to_base64(img):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"

# ------------------------------------------------------------------------------
# STREAMLINED UI
# ------------------------------------------------------------------------------

def setup_ui():
    """Setup streamlined UI for performance"""
    st.set_page_config(page_title="GuardianAI", layout="wide", initial_sidebar_state="expanded")
    
    # Clean, minimal CSS for better aesthetics
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Orbitron', monospace;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Clean header - no decorative bars */
    .clean-header {
        text-align: center;
        margin: 2rem 0 3rem 0;
        padding: 0;
    }
    
    .logo {
        width: 80px;
        height: 80px;
        margin: 0 auto 20px auto;
        position: relative;
        animation: float 3s ease-in-out infinite;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-10px) rotate(5deg); }
    }
    
    /* Enhanced eye with multiple animations - RESTORED! */
    .eye {
        width: 100%;
        height: 100%;
        background: radial-gradient(circle at center, #00ffff 0%, #0080ff 30%, #000033 60%);
        border-radius: 50% 0;
        transform: rotate(45deg);
        box-shadow: 
            0 0 30px #00ffff,
            inset 0 0 30px rgba(0, 255, 255, 0.5);
        animation: pulse-glow 2s ease-in-out infinite;
        position: relative;
        overflow: hidden;
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            box-shadow: 0 0 30px #00ffff, inset 0 0 30px rgba(0, 255, 255, 0.5);
            transform: rotate(45deg) scale(1);
        }
        50% { 
            box-shadow: 0 0 50px #00ffff, inset 0 0 50px rgba(0, 255, 255, 0.8);
            transform: rotate(45deg) scale(1.05);
        }
    }
    
    .eye::before {
        content: "";
        position: absolute;
        width: 40%;
        height: 40%;
        background: radial-gradient(circle, #ffffff 0%, #00ffff 50%, transparent 70%);
        border-radius: 50%;
        top: 30%;
        left: 30%;
        animation: scan 4s ease-in-out infinite;
    }
    
    @keyframes scan {
        0%, 100% { 
            transform: translate(0, 0) scale(1);
            opacity: 1;
        }
        25% { 
            transform: translate(20%, -20%) scale(1.2);
            opacity: 0.8;
        }
        50% { 
            transform: translate(-20%, -20%) scale(0.8);
            opacity: 1;
        }
        75% { 
            transform: translate(-20%, 20%) scale(1.1);
            opacity: 0.9;
        }
    }
    
    /* Add scanning line effect - RESTORED! */
    .eye::after {
        content: "";
        position: absolute;
        width: 2px;
        height: 100%;
        background: linear-gradient(to bottom, transparent, #00ffff, transparent);
        left: 50%;
        top: 0;
        animation: scan-line 3s linear infinite;
        opacity: 0.7;
    }
    
    @keyframes scan-line {
        0% { 
            transform: translateX(-50%) rotate(-45deg) translateY(-100%);
            opacity: 0;
        }
        50% { 
            opacity: 1;
        }
        100% { 
            transform: translateX(-50%) rotate(-45deg) translateY(100%);
            opacity: 0;
        }
    }
    
    /* Clean title */
    .main-title {
        color: #00ffff !important;   
        font-size: 10rem;     
        font-weight: 900;
        margin: 0;
        letter-spacing: 3px;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.3);
    }
    
    .subtitle {
        color: #ffffff;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 10px 0 0 0;
        font-weight: 400;
    }
    
    /* Control buttons */
    .control-section {
        margin: 2rem 0;
        text-align: center;
    }
    
    /* Clean dashboard without decorative elements */
    .dashboard {
        background: rgba(15, 15, 35, 0.8);
        border: 1px solid rgba(0, 255, 255, 0.2);
        border-radius: 15px;
        padding: 25px;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    /* Video container */
    .video-wrapper {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(0, 255, 255, 0.3);
        background: #000;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    }
    
    .video-container {
        background: #000;
        border-radius: 12px;
        overflow: hidden;
        position: relative;
    }
    
    /* Enhanced stats cards */
    .stat-card {
        background: rgba(0, 255, 255, 0.08);
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 15px 0;
        text-align: center;
        transition: all 0.3s ease;
        backdrop-filter: blur(5px);
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 255, 255, 0.15);
        border-color: rgba(0, 255, 255, 0.5);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 900;
        color: #00ffff;
        margin: 0;
        line-height: 1;
        text-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
    }
    
    .stat-label {
        color: rgba(255, 255, 255, 0.7);
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 8px;
        font-weight: 500;
    }
    
    /* Clean alert history */
    .alert-history {
        background: rgba(10, 10, 30, 0.9);
        border: 1px solid rgba(0, 255, 255, 0.25);
        border-radius: 12px;
        padding: 20px;
        margin: 25px 0;
        max-height: 500px;
        overflow-y: auto;
        backdrop-filter: blur(8px);
    }
    
    .alert-history h3 {
        color: #00ffff;
        font-size: 1.3rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid rgba(0, 255, 255, 0.3);
        padding-bottom: 10px;
    }
    .alert-history, .alert-history * {
        color: #e0f7fa !important;  /* Light cyan for all text in alert history */
    }
    
    /* Alert cards */
    .alert-card {
        background: rgba(20, 20, 40, 0.95);
        border-left: 3px solid #ff0040;
        border-radius: 8px;
        padding: 18px;
        margin: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }
    .alert-card, .alert-card * {
        color: #ffffff !important;      /* White text for alert cards */
    }
                
    /* Force Streamlit warning and error boxes to use light text and custom background */
    .stAlert {
        background-color: rgba(40, 0, 0, 0.7) !important; /* Custom dark red background */
        color: #fff !important;                            /* White text */
        border-left: 5px solid #ff0040 !important;         /* Bright red border */
    }
    .stAlert[data-testid="stAlert-warning"] {
        background-color: rgba(40, 40, 0, 0.7) !important; /* Custom dark yellow background */
        color: #fff !important;
        border-left: 5px solid #FFA500 !important;         /* Orange border */
    }
    .stAlert p {
        color: #fff !important;                            /* White text for paragraphs inside alerts */
    }
    /* Force all markdown and text in alert history to be light */
    .alert-history *, .stMarkdown, .stText, .stExpander, .stExpander * {
        color: #e0f7fa !important;
    }
                
    .alert-card:hover {
        transform: translateX(3px);
        box-shadow: 0 4px 15px rgba(255, 0, 64, 0.2);
    }
    
    .alert-card.warning {
        border-left-color: #FFA500;
    }
    
    .alert-card.warning:hover {
        box-shadow: 0 4px 15px rgba(255, 165, 0, 0.2);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 30, 1) !important;
    }

    /* Remove background, border, and filter from sidebar toggle button and its parent */
    button[data-testid="baseButton-headerNoPadding"],
    button[data-testid="baseButton-headerNoPadding"]:hover,
    button[data-testid="baseButton-headerNoPadding"]:focus,
    button[data-testid="baseButton-headerNoPadding"]:active,
    button[data-testid="baseButton-headerNoPadding"] > div,
    button[data-testid="baseButton-headerNoPadding"] > div:hover,
    button[data-testid="baseButton-headerNoPadding"] > div:focus,
    button[data-testid="baseButton-headerNoPadding"] > div:active {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        outline: none !important;
        filter: none !important;
        opacity: 1 !important;
    }

    /* Make the arrow always solid white and fully opaque */
    button[data-testid="baseButton-headerNoPadding"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg path {
        fill: #fff !important;
        stroke: #fff !important;
        opacity: 1 !important;
        filter: none !important;
    }

    /* Remove any filter/opacity from the sidebar itself */
    section[data-testid="stSidebar"] {
        background: rgba(10, 10, 30, 1) !important;
        filter: none !important;
        opacity: 1 !important;
    }
                
    .sidebar-header {
        color: #00ffff;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin: 15px 0 8px 0;
        font-size: 1rem;
    }
                
    label {
        color: #ffffff !important;
    }
    
    /* Input styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: rgba(256, 256, 256, 0.008) !important;
        border: 1px solid rgba(0, 255, 255, 0.3) !important;
        color: #000000 !important;
        border-radius: 6px !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: rgba(0, 255, 255, 0.6) !important;
        box-shadow: 0 0 0 1px rgba(0, 255, 255, 0.3) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: rgba(0, 255, 255, 0.1);
        color: #00ffff;
        border: 1px solid rgba(0, 255, 255, 0.3);
        border-radius: 6px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(0, 255, 255, 0.2);
        border-color: rgba(0, 255, 255, 0.6);
        transform: translateY(-1px);
    }
    
    /* Scrollbar */
    .alert-history::-webkit-scrollbar {
        width: 6px;
    }
    
    .alert-history::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 3px;
    }
    
    .alert-history::-webkit-scrollbar-thumb {
        background: rgba(0, 255, 255, 0.4);
        border-radius: 3px;
    }
    
    .alert-history::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 255, 255, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Clean header without decorative bars
    st.markdown("""
    <div class="clean-header">
        <div class="logo">
            <div class="eye"></div>
        </div>
        <h1 class="main-title">GUARDIAN AI</h1>
        <p class="subtitle">Advanced Multi-Person Fall Detection System</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Control buttons
    st.markdown('<div class="control-section">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    with col2:
        start = st.button("▶️ START MONITORING", use_container_width=True, key="start_btn")
    with col4:
        stop = st.button("⏹️ STOP MONITORING", use_container_width=True, key="stop_btn")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<p class="sidebar-header">⚙️ Configuration</p>', unsafe_allow_html=True)
        
        # Camera settings
        st.markdown('<p class="sidebar-header">📹 Camera Settings</p>', unsafe_allow_html=True)
        input_type = st.selectbox("Input Source", ["Webcam", "IP Camera"])
        
        if input_type == "IP Camera":
            camera_source = st.text_input("Camera URL", "http://172.19.102.161:8080/video")
        else:
            camera_source = st.number_input("Camera Index", 0, 10, 1)  # Default to 1 for iPhone camera
        
        st.markdown("---")
        
        # Alert settings
        st.markdown('<p class="sidebar-header">🚨 Alert Configuration</p>', unsafe_allow_html=True)
        enable_sound = st.checkbox("🔊 Enable Sound Alerts", value=True)
        
        # Sound test buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⚠️ Test Warning", use_container_width=True):
                try:
                    warning_sound.play()
                    st.success("✅ Warning sound!")
                except:
                    st.error("❌ Warning failed!")
        with col2:
            if st.button("🚨 Test Alarm", use_container_width=True):
                try:
                    alarm_sound.play()
                    st.success("✅ Alarm sound!")
                except:
                    st.error("❌ Alarm failed!")
        
        st.markdown("---")
        
        enable_email = st.checkbox("📧 Enable Email Alerts", value=True)
        
        if enable_email:
            st.markdown('<p class="sidebar-header">📧 Email Settings</p>', unsafe_allow_html=True)
            recipient = st.text_input("Recipient Email", DEFAULT_RECIPIENT_EMAIL)
            sender = st.text_input("Your Email Address", "kong.alert.bot24@gmail.com")
            password = st.text_input("Email Password", type="password", value = "mlkhiejdreregbwd")
        else:
            recipient = DEFAULT_RECIPIENT_EMAIL
            sender = ""
            password = ""
        
        st.markdown("---")
        
        st.markdown('<p style="color: rgba(255,255,255,0.4); font-size: 0.75rem; text-align: center; font-weight: 400;">Guardian AI v2.0<br/>Multi-Person Detection</p>', unsafe_allow_html=True)
    
    return (start, stop, camera_source, enable_sound, enable_email,
            recipient, DEFAULT_SMTP_SERVER, DEFAULT_SMTP_PORT, sender, password)

def initialize_session_state():
    """Initialize session state variables"""
    if 'run' not in st.session_state:
        st.session_state.run = False
    if 'processor' not in st.session_state:
        st.session_state.processor = None
    if 'total_falls' not in st.session_state:
        st.session_state.total_falls = 0
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'active_persons' not in st.session_state:
        st.session_state.active_persons = 0

def main():
    """Main application with ultra performance"""
    initialize_session_state()
    
    # Setup UI
    (start, stop, camera_source, enable_sound, enable_email,
     recipient, smtp_server, smtp_port, sender, password) = setup_ui()
    
    # Handle controls
    if start:
        st.session_state.run = True
    if stop:
        st.session_state.run = False
        if st.session_state.processor:
            st.session_state.processor.stop_processing()
    
    # Dashboard layout
    st.markdown('<div class="dashboard">', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2.5, 1])
    
    with col1:
        st.markdown('<div class="video-wrapper"><div class="video-container">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col2:
        # Create placeholders for real-time updates
        active_persons_placeholder = st.empty()
        falls_detected_placeholder = st.empty()
        status_placeholder = st.empty()
        fps_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Alert History Section
    alert_history_placeholder = st.empty()
    
    # Function to update stats display
    def update_stats_display(fps=0):
        # Active persons card
        active_persons_placeholder.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{st.session_state.active_persons}</p>
            <p class="stat-label">Active Persons</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Falls detected card
        falls_detected_placeholder.markdown(f"""
        <div class="stat-card">
            <p class="stat-number">{st.session_state.total_falls}</p>
            <p class="stat-label">Falls Detected</p>
        </div>
        """, unsafe_allow_html=True)
        
        # System status card
        status_color = "#00ff00" if st.session_state.run else "#ff0040"
        status_text = "MONITORING" if st.session_state.run else "STANDBY"
        status_placeholder.markdown(f"""
        <div class="stat-card">
            <p class="stat-number" style="color: {status_color}; font-size: 1.8rem;">{status_text}</p>
            <p class="stat-label">System Status</p>
        </div>
        """, unsafe_allow_html=True)
        
        # FPS display
        fps_placeholder.markdown(f"""
        <div class="stat-card">
            <p class="stat-number" style="font-size: 1.5rem;">{fps:.1f} FPS</p>
            <p class="stat-label">Performance</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Function to update alert history display - NATIVE STREAMLIT COMPONENTS
    def update_alert_history():
        """Update the alert history display with NATIVE STREAMLIT COMPONENTS"""
        
        # Use a container for the alert history
        with alert_history_placeholder.container():
            st.markdown("### 🚨 ALERT HISTORY")
            
            if st.session_state.alerts and len(st.session_state.alerts) > 0:
                # Show last 5 alerts
                recent_alerts = list(reversed(st.session_state.alerts[-5:]))
                
                for alert in recent_alerts:
                    confidence_percent = alert.get('confidence', 0) * 100
                    duration_text = f"{alert.get('duration', 0):.1f}s"
                    alert_type = alert.get('alert_type', 'alarm')
                    
                    icon = "⚠️" if alert_type == 'warning' else "🚨"
                    title = "FALL WARNING" if alert_type == 'warning' else "FALL CONFIRMED"
                    
                    # Use Streamlit's expander for each alert
                    with st.expander(f"{icon} {title} - Person {alert.get('person_id', 'Unknown')}", expanded=True):
                        # Create columns for info and image
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.write(f"🕒 **Time:** {alert.get('timestamp', 'Unknown')}")
                            st.write(f"📊 **Confidence:** {confidence_percent:.1f}%")
                            st.write(f"⏱️ **Duration:** {duration_text}")
                            st.write(f"🎞️ **Frames:** {alert.get('consecutive_frames', 'N/A')}")
                            
                            # Show alert type
                            if alert_type == 'warning':
                                st.warning("⚠️ Fall Warning - Monitoring situation")
                            else:
                                st.error("🚨 Fall Confirmed - Immediate attention required!")
                        
                        with col2:
                            # Display the falling frame image
                            if 'frame_image_base64' in alert and alert['frame_image_base64']:
                                st.markdown("**📷 Captured Frame:**")
                                st.markdown(f'<img src="{alert["frame_image_base64"]}" style="width: 100%; border-radius: 8px; border: 2px solid #00ffff;">', unsafe_allow_html=True)
                            else:
                                st.info("📷 No image available")
                                
            else:
                st.info("🛡️ No alerts detected. Guardian AI is actively monitoring for fall incidents.")
    
    # Initial display update
    update_stats_display()
    update_alert_history()
    
    # Main processing loop
    if st.session_state.run:
        try:
            # Load models if not loaded
            if st.session_state.processor is None:
                with st.spinner("🔄 Loading AI models..."):
                    # Model paths
                    pose_model_path = "inference/yolo11m-pose.pt"
                    action_model_path = "training/BiLSTMWithAttention_best_model_3.0.pth"
                    
                    # Load models
                    pose_model = YOLO(pose_model_path)
                    
                    action_classes = ["Falling", "No Action", "Waving"]
                    action_model = ActionRecognitionBiLSTMWithAttention(
                        34, 256, 4, len(action_classes), 0.5
                    ).to(device)
                    action_model.load_state_dict(torch.load(action_model_path, map_location=device))
                    
                    st.session_state.processor = HighPerformanceVideoProcessor(
                        pose_model, action_model, action_classes, device
                    )
            
            # Open video stream with buffer
            cap = cv2.VideoCapture(int(camera_source) if str(camera_source).isdigit() else camera_source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for low latency
            
            if not cap.isOpened():
                st.error("❌ Cannot open camera feed")
                st.session_state.run = False
                update_stats_display()
                return
            
            # Start processing threads
            st.session_state.processor.start_processing(cap)
            
            # Display loop
            fps_timer = time.time()
            frame_count = 0
            
            while st.session_state.run:
                # Get processed frame
                result = st.session_state.processor.get_display_frame()
                
                if result:
                    annotated_frame, fall_detections = result
                    
                    # Update active persons count
                    st.session_state.active_persons = st.session_state.processor.active_count
                    
                    # Handle fall detections
                    for fall in fall_detections:
                        st.session_state.total_falls += 1
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Convert frame to base64 for HTML display
                        frame_base64 = image_to_base64(fall['frame_image'])
                        
                        # Add to alerts with image
                        alert_data = {
                            'person_id': fall['person_id'],
                            'timestamp': timestamp,
                            'confidence': fall['confidence'],
                            'duration': fall['duration'],
                            'consecutive_frames': fall['consecutive_frames'],
                            'frame_image_base64': frame_base64,
                            'alert_type': fall['alert_type']
                        }
                        
                        # Sound alert based on type
                        if enable_sound:
                            try:
                                if fall['alert_type'] == 'warning':
                                    print(f"⚠️ Playing warning sound for Person {fall['person_id']} - {fall['consecutive_frames']} frames")
                                    warning_sound.play()
                                elif fall['alert_type'] == 'alarm':
                                    print(f"🚨 Playing alarm sound for Person {fall['person_id']} - FALL CONFIRMED!")
                                    alarm_sound.play()
                            except Exception as e:
                                print(f"Sound alert error: {e}")
                        
                        # Email alert only for confirmed falls (alarm type)
                        if enable_email and sender and password and fall['alert_type'] == 'alarm':
                            # Encode image for email
                            _, img_encoded = cv2.imencode('.jpg', fall['frame_image'])
                            
                            future = email_executor.submit(
                                send_email_alert_async,
                                recipient, smtp_server, smtp_port, sender, password,
                                fall['person_id'], st.session_state.total_falls, timestamp,
                                img_encoded.tobytes(), fall['confidence'], 
                                fall['duration'], fall['consecutive_frames']
                            )
                        
                        # Add to visual alert history ONLY if it's a confirmed fall (alarm)
                        if fall['alert_type'] == 'alarm':
                            st.session_state.alerts.append(alert_data)
                            # Update alert history immediately for confirmed alarms
                            update_alert_history()

                
                    #Display frame
                    if annotated_frame is not None:
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, use_container_width=True)
                    
                    # Calculate FPS
                    frame_count += 1
                    if frame_count % 30 == 0:
                        current_time = time.time()
                        fps = 30 / (current_time - fps_timer)
                        fps_timer = current_time
                        update_stats_display(fps)
                    else:
                        # Update other stats without FPS calculation
                        active_persons_placeholder.markdown(f"""
                        <div class="stat-card">
                            <p class="stat-number">{st.session_state.active_persons}</p>
                            <p class="stat-label">Active Persons</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        falls_detected_placeholder.markdown(f"""
                        <div class="stat-card">
                            <p class="stat-number">{st.session_state.total_falls}</p>
                            <p class="stat-label">Falls Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                
            # Cleanup
            st.session_state.processor.stop_processing()
            cap.release()
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.session_state.run = False
            if st.session_state.processor:
                st.session_state.processor.stop_processing()
            update_stats_display()
    else:
        video_placeholder.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; height: 400px; color: #666;">
            <div style="text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 20px;">🛡️</div>
                <p style="font-size: 1.2rem; margin-bottom: 10px;">Guardian AI Ready</p>
                <p>Click START MONITORING to begin fall detection</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Helper function to add sample alerts for testing
def add_sample_alerts():
    """Add sample alerts for testing the alert history display"""
    sample_alerts = [
        {
            'person_id': 3,
            'timestamp': '2025-05-23 23:26:34',
            'confidence': 0.998,
            'duration': 4.9,
            'consecutive_frames': 25,
            'alert_type': 'alarm',
            'frame_image_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAyADIDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q=='
        },
        {
            'person_id': 1,
            'timestamp': '2025-05-23 22:15:12',
            'confidence': 0.85,
            'duration': 2.3,
            'consecutive_frames': 15,
            'alert_type': 'warning',
            'frame_image_base64': 'data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAYEBQYFBAYGBQYHBwYIChAKCgkJChQODwwQFxQYGBcUFhYaHSUfGhsjHBYWICwgIyYnKSopGR8tMC0oMCUoKSj/2wBDAQcHBwoIChMKChMoGhYaKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCgoKCj/wAARCAAyADIDAREAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigAooooAKKKKACiiigD//2Q=='
        }
    ]
    
    # Add sample alerts to session state for testing
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    
    for alert in sample_alerts:
        st.session_state.alerts.append(alert)

if __name__ == "__main__":
    main()