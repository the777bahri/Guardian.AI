import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import time
from collections import deque
import os
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import plotly.graph_objects as go

# --- Streamlit Page Configuration (MUST BE THE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Real-time Action Recognition")

# --- Configuration ---
POSE_MODEL_PATH = "yolov8n-pose.pt"
ACTION_MODEL_PATH = "./models/Guardian_best_model.pth" # Ensure this is in a 'models' subdirectory

pose_model_exists = os.path.exists(POSE_MODEL_PATH)
action_model_exists = os.path.exists(ACTION_MODEL_PATH)

ACTION_CLASSES = ["Falling", "No Action", "Waving"] # Ensure order matches model output
ACTION_ICONS = {
    "Falling": "⚠️",
    "No Action": "🚶", 
    "Waving": "👋",
    "Fall Detected": "🚨" # Icon for the fall counter card
}
ACTION_COLORS = { # Hex colors for card and plotly bars
    "Falling": "#FF4B4B", # Red
    "No Action": "#4CAF50", # Green
    "Waving": "#2196F3",  # Blue
    "Fall Detected": "#FF8C00" # Dark Orange for fall counter card
}


FRAMES_BUFFER_SIZE = 35
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

INPUT_SIZE = 34
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_CLASSES = len(ACTION_CLASSES)
DROPOUT_RATE = 0.5

# --- Performance & Display Configuration ---
PROCESSING_FRAME_WIDTH = 640 
FOCUSED_FRAME_INSET_SIZE = (200, 250) 
FOCUSED_FRAME_BG_COLOR = (30, 30, 30)
PROB_CHART_HEIGHT_INCHES = 3 
PROB_CHART_LINE_WIDTH = 2.5 
DEFAULT_FALL_ALERT_THRESHOLD = 10 # Default consecutive frames for a fall


SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (5, 11), (6, 12)
]

# --- Helper Function to Generate Colored Progress Bar HTML ---
def colored_fall_progress_bar(current, total):
    """
    Generates HTML for a progress bar whose color changes based on the percentage.
    """
    pct = 0.0
    if total > 0: 
        pct = min(1.0, current / total)
    percent_int = int(pct * 100) 

    if pct < 0.5: color = "#4caf50"  # Green
    elif pct < 0.8: color = "#ffc107"  # Amber
    else: color = "#f44336"  # Red

    # Ensure HTML starts immediately without leading spaces from f-string indentation
    bar_html = (
        f'<div style="background-color:#333; border-radius:16px; height:28px; width:100%; box-shadow: 0 1px 4px rgba(0,0,0,0.2) inset; margin-top: 5px;">'
        f'<div style="'
        f'width:{percent_int}%; '
        f'background:{color};'
        f'height:28px; '
        f'border-radius:16px;'
        f'transition: width 0.3s ease-in-out, background-color 0.3s ease-in-out;'
        f'display:flex;'
        f'align-items:center;'
        f'justify-content:center;'
        f'font-weight:bold;'
        f'color:#FFFFFF; ' # White text for better contrast
        f'font-size:0.85em;'
        f'overflow: hidden;'
        f'white-space: nowrap;">'
        f'{current}/{total} ({percent_int}%)'
        f'</div></div>'
    )
    return bar_html

# --- 1. Data Preprocessing (PoseDataset) ---
class PoseDataset(object):
    def __init__(self, max_frames=FRAMES_BUFFER_SIZE):
        self.max_frames = max_frames
        self.feature_size = INPUT_SIZE
    def process_keypoints(self, keypoints_data_sequence):
        all_frames_features = []
        previous_frame_normalized = None
        alpha = 0.8 
        conf_threshold = 0.2 
        relevant_keypoints_data = keypoints_data_sequence[-self.max_frames:]
        for frame_data_dict in relevant_keypoints_data:
            processed_feature_vector = np.zeros(self.feature_size)
            try:
                if not isinstance(frame_data_dict, dict) or 'keypoints' not in frame_data_dict:
                    all_frames_features.append(processed_feature_vector); previous_frame_normalized = None; continue
                person_keypoints_list_of_lists = frame_data_dict['keypoints']
                if not isinstance(person_keypoints_list_of_lists, list) or \
                   len(person_keypoints_list_of_lists) == 0 or \
                   not isinstance(person_keypoints_list_of_lists[0], list) or \
                   len(person_keypoints_list_of_lists[0]) == 0:
                    all_frames_features.append(processed_feature_vector); previous_frame_normalized = None; continue
                frame_keypoints_np = np.array(person_keypoints_list_of_lists[0]).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    all_frames_features.append(processed_feature_vector); previous_frame_normalized = None; continue
                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > conf_threshold]
                if valid_keypoints.shape[0] < 2: 
                    all_frames_features.append(processed_feature_vector); previous_frame_normalized = None; continue
                mean_x = np.mean(valid_keypoints[:, 0]); std_x = np.std(valid_keypoints[:, 0]) + 1e-8 
                mean_y = np.mean(valid_keypoints[:, 1]); std_y = np.std(valid_keypoints[:, 1]) + 1e-8
                normalized_kps = frame_keypoints_np.copy()
                normalized_kps[:, 0] = (normalized_kps[:, 0] - mean_x) / std_x
                normalized_kps[:, 1] = (normalized_kps[:, 1] - mean_y) / std_y
                if previous_frame_normalized is not None: 
                    normalized_kps[:, :2] = alpha * normalized_kps[:, :2] + (1 - alpha) * previous_frame_normalized[:, :2]
                previous_frame_normalized = normalized_kps
                processed_feature_vector = normalized_kps[:, :2].flatten() 
                if processed_feature_vector.shape[0] != self.feature_size:
                    processed_feature_vector = np.zeros(self.feature_size) 
            except Exception as e:
                processed_feature_vector = np.zeros(self.feature_size); previous_frame_normalized = None
            all_frames_features.append(processed_feature_vector)
        num_processed = len(all_frames_features)
        if num_processed == 0: return None
        padded_keypoints = np.zeros((self.max_frames, self.feature_size))
        start_idx_pad = max(0, self.max_frames - num_processed) 
        for i in range(num_processed):
            if all_frames_features[i].shape[0] == self.feature_size:
                 padded_keypoints[start_idx_pad + i, :] = all_frames_features[i]
        if padded_keypoints.shape != (self.max_frames, self.feature_size): return None
        return padded_keypoints

# --- 2. Model Definition ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_output):
        scores = torch.tanh(self.attention_weights(lstm_output))
        attention_weights_softmax = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights_softmax * lstm_output, dim=1)
        return context_vector, attention_weights_softmax.squeeze(-1)

class ActionRecognitionBiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(ActionRecognitionBiLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.attention = AttentionLayer(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0)); lstm_out = self.dropout(lstm_out)
        context_vector, attention_weights = self.attention(lstm_out)
        context_vector_dropped = self.dropout(context_vector)
        logits = self.fc(context_vector_dropped)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities, attention_weights

# --- 3. Load Models ---
@st.cache_resource
def load_pose_model(path):
    try: model = YOLO(path); return model
    except Exception as e: st.error(f"Error loading YOLO pose model from {path}: {e}"); return None
@st.cache_resource
def load_action_model(_action_model_path, _device):
    model = ActionRecognitionBiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(_device)
    try:
        model.load_state_dict(torch.load(_action_model_path, map_location=_device)); model.eval()
        st.success(f"Action recognition model loaded successfully from {_action_model_path}.")
        return model
    except FileNotFoundError: st.error(f"Error: Action model file not found at {_action_model_path}"); return None
    except Exception as e: st.error(f"Error loading action model state_dict from {_action_model_path}: {e}"); return None

models_ready = True
if not pose_model_exists: st.error(f"Pose model not found at {POSE_MODEL_PATH}."); models_ready = False
if not action_model_exists: st.error(f"Action model not found at {ACTION_MODEL_PATH}."); models_ready = False
pose_model, action_model = None, None
if models_ready:
    pose_model = load_pose_model(POSE_MODEL_PATH)
    action_model = load_action_model(ACTION_MODEL_PATH, DEVICE)
    if pose_model is None or action_model is None: models_ready = False
else: st.warning("One or more models could not be loaded."); st.stop()
pose_preprocessor = PoseDataset(max_frames=FRAMES_BUFFER_SIZE)

# --- 4. Prediction Function ---
def predict_action(model, keypoints_sequence_buffer):
    if model is None:
        st.warning("Action model not loaded.")
        default_probs = np.zeros(len(ACTION_CLASSES))
        no_action_idx = ACTION_CLASSES.index("No Action") if "No Action" in ACTION_CLASSES else 0
        default_probs[no_action_idx] = 1.0
        return ACTION_CLASSES[no_action_idx], default_probs, np.zeros(FRAMES_BUFFER_SIZE)
    model.eval()
    normalized_keypoints = pose_preprocessor.process_keypoints(list(keypoints_sequence_buffer))
    default_probs = np.zeros(len(ACTION_CLASSES)); default_attention = np.zeros(FRAMES_BUFFER_SIZE)
    no_action_idx = ACTION_CLASSES.index("No Action") if "No Action" in ACTION_CLASSES else 0
    default_probs[no_action_idx] = 1.0
    if normalized_keypoints is None: return ACTION_CLASSES[no_action_idx], default_probs, default_attention
    if normalized_keypoints.shape != (FRAMES_BUFFER_SIZE, INPUT_SIZE): return ACTION_CLASSES[no_action_idx], default_probs, default_attention
    normalized_keypoints_tensor = torch.tensor(normalized_keypoints, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities_tensor, attention_weights_tensor = model(normalized_keypoints_tensor)
        probabilities = probabilities_tensor.cpu().numpy()[0]
        attention_weights = attention_weights_tensor.cpu().numpy()[0]
    predicted_index = np.argmax(probabilities)
    predicted_action_label = ACTION_CLASSES[predicted_index]
    return predicted_action_label, probabilities, attention_weights

# --- 5. Visualization Functions ---
def draw_skeleton(canvas, keypoints_xyc, connections, color=(255, 0, 0), thickness=2):
    if keypoints_xyc is None: return
    plotted_points = []
    for i in range(len(keypoints_xyc)):
        x_scaled, y_scaled, conf = keypoints_xyc[i]
        if x_scaled is not None and y_scaled is not None and conf > 0.1:
            cv2.circle(canvas, (int(x_scaled), int(y_scaled)), thickness + 1, color, -1)
            plotted_points.append((int(x_scaled), int(y_scaled)))
        else: plotted_points.append(None)
    for i, (p1_idx, p2_idx) in enumerate(connections):
        if p1_idx < len(plotted_points) and p2_idx < len(plotted_points) and \
           plotted_points[p1_idx] is not None and plotted_points[p2_idx] is not None:
            cv2.line(canvas, plotted_points[p1_idx], plotted_points[p2_idx], color, thickness)

def create_focused_skeleton_image(raw_keypoints_single_person, size, bg_color, connections):
    inset_w, inset_h = size
    inset_canvas = np.full((inset_h, inset_w, 3), bg_color, dtype=np.uint8)
    if not raw_keypoints_single_person: return inset_canvas
    valid_pts = [pt for pt in raw_keypoints_single_person if pt[2] > 0.1]
    scaled_kpts = []
    if len(valid_pts) >= 1:
        xs, ys = [pt[0] for pt in valid_pts], [pt[1] for pt in valid_pts]
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        kpt_bbox_w, kpt_bbox_h = max_x - min_x, max_y - min_y
        if kpt_bbox_w == 0 and kpt_bbox_h == 0: 
            for i in range(len(raw_keypoints_single_person)):
                scaled_kpts.append((inset_w // 2, inset_h // 2, raw_keypoints_single_person[i][2]) \
                                   if i < len(valid_pts) and raw_keypoints_single_person[i][2] > 0.1 else (None, None, 0))
        elif kpt_bbox_w > 0 or kpt_bbox_h > 0: 
            scale_x = (inset_w * 0.8) / kpt_bbox_w if kpt_bbox_w > 0 else 1
            scale_y = (inset_h * 0.8) / kpt_bbox_h if kpt_bbox_h > 0 else 1
            scale = min(scale_x, scale_y)
            for x_orig, y_orig, v_orig in raw_keypoints_single_person:
                if v_orig > 0.1:
                    nx = int((x_orig - min_x) * scale + (inset_w - kpt_bbox_w * scale) / 2)
                    ny = int((y_orig - min_y) * scale + (inset_h - kpt_bbox_h * scale) / 2)
                    scaled_kpts.append((nx, ny, v_orig))
                else: scaled_kpts.append((None, None, v_orig))
        else: 
            scaled_kpts = [(None,None,0)] * len(raw_keypoints_single_person)
        draw_skeleton(inset_canvas, scaled_kpts, connections, color=(0, 255, 255), thickness=1)
    return inset_canvas

def create_probability_trend_chart_matplotlib(prob_df, action_classes_list, fig_height_inches=3, line_width=2.0):
    if prob_df.empty:
        fig, ax = plt.subplots(figsize=(10, fig_height_inches)) 
        ax.text(0.5, 0.5, "Waiting for probability data...", ha='center', va='center', fontsize=9); ax.axis('off')
    else:
        fig, ax = plt.subplots(figsize=(10, fig_height_inches)); plt.style.use('seaborn-v0_8-darkgrid') 
        plot_colors = [ACTION_COLORS.get(ac, "#888888") for ac in action_classes_list] 
        for i, col in enumerate(action_classes_list):
            if col in prob_df.columns:
                ax.plot(prob_df.index, prob_df[col], label=col, color=plot_colors[i], linewidth=line_width)
        ax.set_xlabel("Frame Number (Buffer Window)", fontsize=8); ax.set_ylabel("Probability", fontsize=8)
        ax.set_ylim(0, 1.1); ax.legend(loc='upper left', fontsize='x-small')
        ax.tick_params(axis='both', which='major', labelsize=7); fig.tight_layout(pad=0.3)
    buf = BytesIO(); fig.savefig(buf, format="png", dpi=120); plt.close(fig); buf.seek(0)
    img_arr = np.array(plt.imread(buf, format='png'))
    if img_arr.shape[2] == 4: img_arr = img_arr[:, :, :3] 
    return (img_arr * 255).astype(np.uint8)

def update_action_card_html(action_name, confidence_score, card_title="Current Action", icon_override=None, color_override=None):
    icon = icon_override if icon_override else ACTION_ICONS.get(action_name, "❓")
    color = color_override if color_override else ACTION_COLORS.get(action_name, "#888888") 
    html = (
        f'<div style="border:2px solid {color};border-radius:10px;padding:10px;text-align:center;background-color:#222;margin-bottom:5px;box-shadow:0 4px 8px 0 rgba(0,0,0,0.3);">'
        f'<span style="font-size:1.0em;font-weight:bold;color:#FFFFFF;display:block;margin-bottom:5px;">{card_title}</span>'
        f'<span style="font-size:2.0em;margin-right:8px;">{icon}</span>'
        f'<div style="margin-top:3px;display:inline-block;vertical-align:middle;">'
        f'<span style="font-size:1.3em;font-weight:bold;color:{color};">{action_name}</span><br>'
        f'<span style="font-size:1.0em;color:#FFFFFF;">Confidence: {(confidence_score*100):.1f}%</span></div></div>'
    )
    return html

def update_fall_detection_card_html(total_falls, current_consecutive, threshold, is_falling_active):
    """Generates HTML for the combined fall detection status card."""
    icon = ACTION_ICONS.get("Fall Detected", "🚨")
    color = ACTION_COLORS.get("Fall Detected", "#FF8C00")
    
    progress_text_html = ""
    if is_falling_active:
        pct_fall_display = 0
        if threshold > 0:
            pct_fall_display = int((current_consecutive / threshold) * 100)
        progress_text_html = ( # Ensure no leading spaces in this multi-line f-string part
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;margin-bottom:-5px;">'
            f'<span style="font-size:0.9em;font-weight:600;color:#DDD;">Fall Progress:</span>'
            f'<span style="font-size:0.9em;font-weight:bold;color:#DDD;">{current_consecutive}/{threshold} ({pct_fall_display}%)</span>'
            f'</div>'
        )
    else:
        progress_text_html = ( # Ensure no leading spaces
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;margin-bottom:-5px;">'
            f'<span style="font-size:0.9em;font-weight:600;color:#888;">Fall Progress:</span>'
            f'<span style="font-size:0.9em;font-weight:bold;color:#888;">Not Actively Falling</span>'
            f'</div>'
        )

    progress_bar_html_content = colored_fall_progress_bar(current_consecutive if is_falling_active else 0, threshold)

    html_card_top = (
        f'<div style="border:2px solid {color};border-radius:10px;padding:10px;text-align:center;background-color:#2A2A2A;margin-bottom:5px;box-shadow:0 4px 8px 0 rgba(0,0,0,0.3);">'
        f'<span style="font-size:1.1em;font-weight:bold;color:#FFFFFF;display:block;margin-bottom:5px;">Fall Detection System</span>'
        f'<div style="display:flex;align-items:center;justify-content:center;margin-bottom:5px;">'
        f'<span style="font-size:2.2em;margin-right:10px;color:{color};">{icon}</span>'
        f'<span style="font-size:1.4em;font-weight:bold;color:{color};">Total Falls: {total_falls}</span></div>'
    )
    html_card_bottom = f'</div>'
    
    return html_card_top + progress_text_html + progress_bar_html_content + html_card_bottom


def create_confidence_bars_plotly(probabilities_array, action_classes_list):
    bar_colors = [ACTION_COLORS.get(ac, "#888888") for ac in action_classes_list]
    fig = go.Figure(go.Bar(y=action_classes_list, x=probabilities_array, orientation='h', marker_color=bar_colors,
                           text=[f"{(p*100):.1f}%" for p in probabilities_array], textposition='outside'))
    fig.update_layout(title_text='Confidence Distribution', title_x=0.25, title_y=0.95, # Ensure title has space
                      xaxis_title=None, yaxis_title=None, yaxis_autorange="reversed", 
                      height=180 + len(action_classes_list) * 25, # Increased height for better fit
                      margin=dict(l=5, r=5, t=40, b=10), # Increased top margin for title
                      paper_bgcolor='rgba(0,0,0,0)', 
                      plot_bgcolor='rgba(0,0,0,0)', font=dict(color="#FFFFFF", size=10), 
                      xaxis=dict(range=[0, 1], showgrid=True, gridcolor='rgba(255,255,255,0.1)', tickfont=dict(size=9)),
                      yaxis=dict(showgrid=False, tickfont=dict(size=10)))
    return fig


# --- Streamlit App UI ---
st.title("🏋️ Real-time Action Recognition & Fall Detection") 
st.markdown("This application performs real-time human action recognition using YOLO and a BiLSTM with Attention model, including fall detection.")

# Initialize session state variables
if 'run_processing' not in st.session_state: st.session_state.run_processing = False 
if 'input_source_type' not in st.session_state: st.session_state.input_source_type = "Webcam"
if 'stream_url' not in st.session_state: st.session_state.stream_url = ""
if 'keypoint_buffer' not in st.session_state: st.session_state.keypoint_buffer = deque(maxlen=FRAMES_BUFFER_SIZE)
if 'raw_keypoints_buffer_for_viz' not in st.session_state: st.session_state.raw_keypoints_buffer_for_viz = deque(maxlen=FRAMES_BUFFER_SIZE)
if 'last_probabilities' not in st.session_state:
    st.session_state.last_probabilities = np.zeros(NUM_CLASSES)
    no_action_idx = ACTION_CLASSES.index("No Action") if "No Action" in ACTION_CLASSES else 0
    st.session_state.last_probabilities[no_action_idx] = 1.0
if 'last_attention_weights' not in st.session_state: st.session_state.last_attention_weights = np.zeros(FRAMES_BUFFER_SIZE)
if 'predicted_action' not in st.session_state: st.session_state.predicted_action = "No Action"
if 'frame_count' not in st.session_state: st.session_state.frame_count = 0
if 'prob_history_df' not in st.session_state: st.session_state.prob_history_df = pd.DataFrame(columns=['Frame'] + ACTION_CLASSES).set_index('Frame')

if 'consecutive_fall_frames' not in st.session_state: st.session_state.consecutive_fall_frames = 0
if 'fall_alert_triggered' not in st.session_state: st.session_state.fall_alert_triggered = False
if 'total_falls_detected' not in st.session_state: st.session_state.total_falls_detected = 0
if 'fall_alert_threshold' not in st.session_state: st.session_state.fall_alert_threshold = DEFAULT_FALL_ALERT_THRESHOLD


# Sidebar controls
st.sidebar.header("Controls")
st.session_state.input_source_type = st.sidebar.radio(
    "Select Input Source", ("Webcam", "Video Stream URL"),
    index=("Webcam", "Video Stream URL").index(st.session_state.input_source_type)) 

if st.session_state.input_source_type == "Video Stream URL":
    st.session_state.stream_url = st.sidebar.text_input("Enter Video Stream URL (RTSP, HTTP, etc.):", st.session_state.stream_url)

yolo_conf_threshold = st.sidebar.slider("YOLO Confidence", 0.1, 1.0, 0.5, 0.05)
st.session_state.fall_alert_threshold = st.sidebar.slider(
    "Fall Alert Threshold (Consecutive Frames)", min_value=3, max_value=30, 
    value=st.session_state.fall_alert_threshold, step=1)

if st.sidebar.button("Start Processing", key="start_processing"): st.session_state.run_processing = True
if st.sidebar.button("Stop Processing", key="stop_processing"):  
    st.session_state.run_processing = False
    st.session_state.keypoint_buffer.clear(); st.session_state.raw_keypoints_buffer_for_viz.clear()
    st.session_state.prob_history_df = pd.DataFrame(columns=['Frame'] + ACTION_CLASSES).set_index('Frame')
    st.session_state.predicted_action = "No Action"
    st.session_state.last_probabilities = np.zeros(NUM_CLASSES)
    no_action_idx = ACTION_CLASSES.index("No Action") if "No Action" in ACTION_CLASSES else 0
    st.session_state.last_probabilities[no_action_idx] = 1.0
    st.session_state.last_attention_weights = np.zeros(FRAMES_BUFFER_SIZE)
    st.session_state.consecutive_fall_frames = 0 
    st.session_state.fall_alert_triggered = False

# --- Main Display Area ---
top_col1, top_col2 = st.columns([3, 1.2]) 

with top_col1:
    st.subheader("Live Video Feed")
    FRAME_WINDOW = st.image([])
    fps_placeholder = st.empty()

with top_col2:
    action_card_placeholder = st.empty()
    action_card_placeholder.markdown(update_action_card_html("Initializing...", 0.0), unsafe_allow_html=True)
    
    confidence_bars_placeholder = st.empty() 
    confidence_bars_placeholder.plotly_chart(create_confidence_bars_plotly(st.session_state.last_probabilities, ACTION_CLASSES), use_container_width=True)

    # Unified Fall Detection Status Card
    fall_detection_status_placeholder = st.empty()
    fall_detection_status_placeholder.markdown(
        update_fall_detection_card_html(
            st.session_state.total_falls_detected, 
            0, 
            st.session_state.fall_alert_threshold,
            False 
        ), 
        unsafe_allow_html=True
    )

mid_col1, mid_col2 = st.columns([3, 1.2]) 

with mid_col1:
    st.subheader("Action Probability Trend")
    prob_chart_placeholder = st.image([]) 
    prob_chart_placeholder.image(create_probability_trend_chart_matplotlib(st.session_state.prob_history_df, ACTION_CLASSES, PROB_CHART_HEIGHT_INCHES, PROB_CHART_LINE_WIDTH), use_container_width=True)

with mid_col2:
    st.subheader("Focused Frame")
    focused_frame_placeholder = st.image([])
    focused_frame_placeholder.image(np.full((FOCUSED_FRAME_INSET_SIZE[1], FOCUSED_FRAME_INSET_SIZE[0], 3), FOCUSED_FRAME_BG_COLOR, dtype=np.uint8), width=FOCUSED_FRAME_INSET_SIZE[0])


st.subheader("Attention Weight Distribution (Frames in Buffer)")
attention_weights_chart_placeholder = st.empty() 
attention_weights_chart_placeholder.bar_chart(np.zeros(FRAMES_BUFFER_SIZE), height=150)


cap = None
if st.session_state.run_processing:
    source_to_use = 0 
    if st.session_state.input_source_type == "Video Stream URL":
        if st.session_state.stream_url: source_to_use = st.session_state.stream_url
        else: st.warning("Video Stream URL is selected but no URL is provided. Using Webcam.")
    try:
        cap = cv2.VideoCapture(source_to_use)
        if not cap.isOpened():
            st.error(f"Cannot open video source: {source_to_use}. Please check the URL or webcam.")
            st.session_state.run_processing = False
    except Exception as e:
        st.error(f"Error opening video source {source_to_use}: {e}"); st.session_state.run_processing = False

loop_start_time = time.time()

while st.session_state.run_processing and cap and cap.isOpened() and models_ready:
    ret, frame_orig = cap.read()
    if not ret: 
        if st.session_state.input_source_type == "Video Stream URL": st.warning("Stream ended or cannot receive frame. Stopping.")
        else: st.warning("Can't receive frame from webcam. Stopping.")
        st.session_state.run_processing = False; break

    h_orig, w_orig = frame_orig.shape[:2]
    processing_frame = cv2.resize(frame_orig, (PROCESSING_FRAME_WIDTH, int(PROCESSING_FRAME_WIDTH * h_orig / w_orig)), interpolation=cv2.INTER_AREA) if w_orig > PROCESSING_FRAME_WIDTH else frame_orig.copy()
    results = pose_model(processing_frame, verbose=False, conf=yolo_conf_threshold)
    annotated_frame = results[0].plot() 
    current_raw_keypoints_for_viz, current_frame_keypoints_for_buffer = None, []
    try:
        if len(results) > 0 and results[0].keypoints is not None and results[0].keypoints.data.numel() > 0:
            person_keypoints_np = results[0].keypoints.data[0].cpu().numpy()
            current_raw_keypoints_for_viz = person_keypoints_np.tolist()
            current_frame_keypoints_for_buffer = [person_keypoints_np.tolist()]
    except Exception as e: pass 
    st.session_state.keypoint_buffer.append({"frame_num": st.session_state.frame_count, "keypoints": current_frame_keypoints_for_buffer})
    st.session_state.raw_keypoints_buffer_for_viz.append(current_raw_keypoints_for_viz)

    if len(st.session_state.keypoint_buffer) == FRAMES_BUFFER_SIZE:
        pred_action, current_probs, current_attn = predict_action(action_model, st.session_state.keypoint_buffer)
        st.session_state.predicted_action = pred_action
        st.session_state.last_probabilities = current_probs 
        st.session_state.last_attention_weights = current_attn
        
        action_card_placeholder.markdown(update_action_card_html(pred_action, current_probs[ACTION_CLASSES.index(pred_action)]), unsafe_allow_html=True)
        confidence_bars_placeholder.plotly_chart(create_confidence_bars_plotly(current_probs, ACTION_CLASSES), use_container_width=True)

        is_falling_active_now = False
        if pred_action == "Falling":
            st.session_state.consecutive_fall_frames += 1
            is_falling_active_now = True
            if st.session_state.consecutive_fall_frames >= st.session_state.fall_alert_threshold and \
               not st.session_state.fall_alert_triggered:
                st.session_state.total_falls_detected += 1
                st.session_state.fall_alert_triggered = True
                cv2.putText(annotated_frame, "FALL DETECTED!", (annotated_frame.shape[1]//2 - 150, annotated_frame.shape[0]//2), 
                            cv2.FONT_HERSHEY_TRIPLEX, 1.5, (0,0,255), 3, cv2.LINE_AA)
        else:
            st.session_state.consecutive_fall_frames = 0
            st.session_state.fall_alert_triggered = False
        
        fall_detection_status_placeholder.markdown( # This now updates the unified card
            update_fall_detection_card_html(
                st.session_state.total_falls_detected, 
                st.session_state.consecutive_fall_frames, 
                st.session_state.fall_alert_threshold,
                is_falling_active_now 
            ), unsafe_allow_html=True)
        
        new_row_data = {'Frame': st.session_state.frame_count}
        for i, cls_name in enumerate(ACTION_CLASSES): new_row_data[cls_name] = current_probs[i]
        new_df_row = pd.DataFrame([new_row_data]).set_index('Frame')
        st.session_state.prob_history_df = pd.concat([st.session_state.prob_history_df, new_df_row])
        if len(st.session_state.prob_history_df) > FRAMES_BUFFER_SIZE*2: 
            st.session_state.prob_history_df = st.session_state.prob_history_df.iloc[-(FRAMES_BUFFER_SIZE*2):]
        if not st.session_state.prob_history_df.empty:
            prob_chart_img = create_probability_trend_chart_matplotlib(
                st.session_state.prob_history_df.tail(FRAMES_BUFFER_SIZE), ACTION_CLASSES,
                PROB_CHART_HEIGHT_INCHES, PROB_CHART_LINE_WIDTH)
            prob_chart_placeholder.image(prob_chart_img, use_container_width=True)
        if st.session_state.last_attention_weights is not None and len(st.session_state.last_attention_weights) > 0 :
            attention_weights_chart_placeholder.bar_chart(st.session_state.last_attention_weights, height=150) 
        else: attention_weights_chart_placeholder.bar_chart(np.zeros(FRAMES_BUFFER_SIZE), height=150)

    if len(st.session_state.keypoint_buffer) == FRAMES_BUFFER_SIZE and \
       st.session_state.last_attention_weights is not None and np.any(st.session_state.last_attention_weights > 1e-3):
        focused_idx = np.argmax(st.session_state.last_attention_weights)
        if len(st.session_state.raw_keypoints_buffer_for_viz) == FRAMES_BUFFER_SIZE:
            kpts_focused = list(st.session_state.raw_keypoints_buffer_for_viz)[focused_idx]
            if kpts_focused:
                focused_frame_placeholder.image(create_focused_skeleton_image(kpts_focused, FOCUSED_FRAME_INSET_SIZE,
                                                FOCUSED_FRAME_BG_COLOR, SKELETON_CONNECTIONS), channels="BGR", width=FOCUSED_FRAME_INSET_SIZE[0])

    cv2.putText(annotated_frame, f"Action: {st.session_state.predicted_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    FRAME_WINDOW.image(annotated_frame, channels="BGR", use_container_width=True) 
    loop_time_elapsed = time.time() - loop_start_time
    current_fps = 1.0 / loop_time_elapsed if loop_time_elapsed > 0 else 0
    loop_start_time = time.time()
    fps_placeholder.caption(f"Processing FPS: {current_fps:.1f}")
    st.session_state.frame_count += 1
else: 
    if cap is not None and cap.isOpened(): cap.release()
    if not st.session_state.run_processing and models_ready:
        FRAME_WINDOW.empty(); fps_placeholder.empty()
        action_card_placeholder.markdown(update_action_card_html("Stopped", 0.0), unsafe_allow_html=True)
        confidence_bars_placeholder.plotly_chart(create_confidence_bars_plotly(np.zeros(NUM_CLASSES), ACTION_CLASSES), use_container_width=True)
        fall_detection_status_placeholder.markdown( # Update unified card when stopped
            update_fall_detection_card_html(
                st.session_state.total_falls_detected, 0, 
                st.session_state.fall_alert_threshold, False
            ), unsafe_allow_html=True) 
        prob_chart_placeholder.image(create_probability_trend_chart_matplotlib(pd.DataFrame(), ACTION_CLASSES, PROB_CHART_HEIGHT_INCHES, PROB_CHART_LINE_WIDTH), use_container_width=True)
        attention_weights_chart_placeholder.bar_chart(np.zeros(FRAMES_BUFFER_SIZE), height=150)
        focused_frame_placeholder.image(np.full((FOCUSED_FRAME_INSET_SIZE[1], FOCUSED_FRAME_INSET_SIZE[0], 3), FOCUSED_FRAME_BG_COLOR, dtype=np.uint8), width=FOCUSED_FRAME_INSET_SIZE[0])
    elif not models_ready: 
        action_card_placeholder.error("App cannot start due to model loading issues.") 

if cap is not None and cap.isOpened(): cap.release() 
st.sidebar.markdown("---")
st.sidebar.info(f"Device: {DEVICE}"); st.sidebar.info(f"Pose Model: {os.path.basename(POSE_MODEL_PATH)}")
st.sidebar.info(f"Action Model: {os.path.basename(ACTION_MODEL_PATH)}"); st.sidebar.info(f"Buffer: {FRAMES_BUFFER_SIZE} frames")
st.sidebar.info(f"Processing Width: {PROCESSING_FRAME_WIDTH}px")

