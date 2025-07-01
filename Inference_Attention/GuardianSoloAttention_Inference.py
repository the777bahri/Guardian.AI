import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" # For potential Matplotlib/torch conflict
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from collections import deque

# --- Configuration ---
POSE_MODEL_PATH = "yolo11n-pose.pt" # Or "yolov8n-pose.pt" if that's what you use
ACTION_MODEL_PATH = "./models/Guardian_best_model.pth" # Ensure this path is correct
ACTION_CLASSES = ["Falling", "No Action", "Waving"]
FRAMES_BUFFER_SIZE = 35
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Action Model Hyperparameters
INPUT_SIZE = 34
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_CLASSES = len(ACTION_CLASSES)
DROPOUT_RATE = 0.5

# Display settings
DISPLAY_SCALE = 1.00
PLOT_UPDATE_INTERVAL = 10 # For Matplotlib probability plot

# Attention Visualization settings
ATTENTION_VIZ_X_START = 10
ATTENTION_VIZ_Y_START = 150 # Y-coordinate for the bottom of the attention bars
ATTENTION_VIZ_HEIGHT = 30   # Max height of an attention bar
ATTENTION_VIZ_BAR_WIDTH_TOTAL = 35 * 6 # Total width for 35 bars

# Focused Frame Visualization settings
FOCUSED_FRAME_INSET_X = 10
FOCUSED_FRAME_INSET_Y = ATTENTION_VIZ_Y_START + ATTENTION_VIZ_HEIGHT + 20 # Below attention bars
FOCUSED_FRAME_INSET_SIZE = (150, 200) # Width, Height for the inset
FOCUSED_FRAME_BG_COLOR = (30, 30, 30)

print(f"Using device: {DEVICE}")

# --- 1. Data Preprocessing (PoseDataset) ---
class PoseDataset(object):
    def __init__(self, data_dir, action_classes, max_frames=FRAMES_BUFFER_SIZE): # Use global constant
        self.data_dir = data_dir # Not used in this inference script's process_keypoints
        self.action_classes = action_classes # Not used in this inference script's process_keypoints
        self.max_frames = max_frames
        self.feature_size = INPUT_SIZE # Use global constant

    def process_keypoints(self, keypoints_data_sequence): # Renamed for clarity
        all_frames_features = []
        previous_frame_normalized = None
        alpha = 0.8
        conf_threshold = 0.2
        
        # keypoints_data_sequence is a list of dicts: [{'keypoints': [[[kpts_person1_frameN]]]}, ...]
        relevant_keypoints_data = keypoints_data_sequence[-self.max_frames:]

        for frame_data_dict in relevant_keypoints_data:
            processed_feature_vector = np.zeros(self.feature_size)
            try:
                if not isinstance(frame_data_dict, dict) or 'keypoints' not in frame_data_dict:
                    all_frames_features.append(processed_feature_vector)
                    previous_frame_normalized = None # Reset EMA state
                    continue
                
                # 'keypoints' from buffer is expected to be like [[[x,y,c]*17]] or []
                person_keypoints_list_of_lists = frame_data_dict['keypoints']

                if not isinstance(person_keypoints_list_of_lists, list) or \
                   len(person_keypoints_list_of_lists) == 0 or \
                   not isinstance(person_keypoints_list_of_lists[0], list) or \
                   len(person_keypoints_list_of_lists[0]) == 0:
                    all_frames_features.append(processed_feature_vector)
                    previous_frame_normalized = None
                    continue
                
                # Get the first (and only for this script) person's keypoints
                frame_keypoints_np = np.array(person_keypoints_list_of_lists[0]).reshape(-1, 3)

                if frame_keypoints_np.shape != (17, 3):
                    all_frames_features.append(processed_feature_vector)
                    previous_frame_normalized = None
                    continue
                
                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > conf_threshold]
                if valid_keypoints.shape[0] < 2:
                    all_frames_features.append(processed_feature_vector)
                    previous_frame_normalized = None
                    continue

                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8

                normalized_kps = frame_keypoints_np.copy()
                normalized_kps[:, 0] = (normalized_kps[:, 0] - mean_x) / std_x
                normalized_kps[:, 1] = (normalized_kps[:, 1] - mean_y) / std_y

                if previous_frame_normalized is not None:
                    normalized_kps[:, :2] = alpha * normalized_kps[:, :2] + \
                                           (1 - alpha) * previous_frame_normalized[:, :2]
                previous_frame_normalized = normalized_kps
                
                processed_feature_vector = normalized_kps[:, :2].flatten()
                if processed_feature_vector.shape[0] != self.feature_size:
                    processed_feature_vector = np.zeros(self.feature_size)
            
            except Exception as e:
                # print(f"Error processing frame keypoints in PoseDataset: {e}")
                processed_feature_vector = np.zeros(self.feature_size)
                previous_frame_normalized = None
            
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
        # print(f"DEBUG AttentionLayer: scores BEFORE softmax = {scores.squeeze().detach().cpu().numpy()}")
        attention_weights_softmax_out = torch.softmax(scores, dim=1)
        # print(f"DEBUG AttentionLayer: attention_weights AFTER softmax (sum={torch.sum(attention_weights_softmax_out, dim=1).item()}): {attention_weights_softmax_out.squeeze().detach().cpu().numpy()}")
        context_vector = torch.sum(attention_weights_softmax_out * lstm_output, dim=1)
        return context_vector, attention_weights_softmax_out.squeeze(-1)

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
        # print(f"DEBUG model: lstm_out sum = {torch.sum(lstm_out)}")
        lstm_out = self.dropout(lstm_out)
        context_vector, attention_weights = self.attention(lstm_out)
        context_vector_dropped = self.dropout(context_vector)
        logits = self.fc(context_vector_dropped)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities, attention_weights

# --- 3. Load Models ---
pose_model = YOLO(POSE_MODEL_PATH)
action_model = ActionRecognitionBiLSTMWithAttention(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE).to(DEVICE)
try:
    action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
    print("Action recognition model loaded successfully.")
    action_model.eval()
except FileNotFoundError:
    print(f"Error: Action model file not found at {ACTION_MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading action model state_dict: {e}")
    exit()

# --- 4. Prediction Function ---
pose_preprocessor = PoseDataset(data_dir=None, action_classes=None, max_frames=FRAMES_BUFFER_SIZE) # Instantiated once

def predict_action(model, keypoints_sequence): # Removed redundant params
    model.eval()
    normalized_keypoints = pose_preprocessor.process_keypoints(keypoints_sequence)

    default_probs = np.zeros(len(ACTION_CLASSES))
    default_attention = np.zeros(FRAMES_BUFFER_SIZE)
    no_action_idx = ACTION_CLASSES.index("No Action") if "No Action" in ACTION_CLASSES else 0
    default_probs[no_action_idx] = 1.0

    if normalized_keypoints is None:
        # print("DEBUG: predict_action returning default due to normalized_keypoints is None")
        return ACTION_CLASSES[no_action_idx], default_probs, default_attention
    if normalized_keypoints.shape != (FRAMES_BUFFER_SIZE, INPUT_SIZE):
        # print(f"DEBUG: predict_action returning default due to shape mismatch. Expected: {(FRAMES_BUFFER_SIZE, INPUT_SIZE)}, Got: {normalized_keypoints.shape}")
        return ACTION_CLASSES[no_action_idx], default_probs, default_attention

    normalized_keypoints_tensor = torch.tensor(normalized_keypoints, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities_tensor, attention_weights_tensor = model(normalized_keypoints_tensor)
        probabilities = probabilities_tensor.cpu().numpy()[0]
        attention_weights = attention_weights_tensor.cpu().numpy()[0]

    predicted_index = np.argmax(probabilities)
    predicted_action = ACTION_CLASSES[predicted_index]
    return predicted_action, probabilities, attention_weights

# --- 5. Visualization Functions ---
def draw_attention_bars(frame, attention_weights, x_start, y_start_bottom, bar_width_total, bar_height_max, bar_color=(0,165,255)):
    if attention_weights is None or len(attention_weights) == 0:
        return
    num_weights = len(attention_weights)
    bar_width_individual = bar_width_total // num_weights
    if bar_width_individual == 0: bar_width_individual = 1

    for i, weight in enumerate(attention_weights):
        bar_h = int(np.clip(weight * bar_height_max * 10, 0, bar_height_max)) # Scale weights for visibility, clip
        cv2.rectangle(frame,
                      (x_start + i * bar_width_individual, y_start_bottom - bar_h),
                      (x_start + (i + 1) * bar_width_individual - 1, y_start_bottom),
                      bar_color,
                      -1)
    cv2.rectangle(frame, (x_start, y_start_bottom - bar_height_max), (x_start + bar_width_total, y_start_bottom), (200,200,200), 1)



def draw_skeleton(canvas, keypoints, color=(255, 0, 0), thickness=2):
    """Draws a skeleton on the given canvas.
    Assumes keypoints are already scaled appropriately for this canvas."""
    if keypoints is None: # Simpler check now
        return
    
    # h, w = canvas.shape[:2] # Not needed if not re-scaling, but good for bounds checks if you add them

    # COCO keypoint connections (ensure this is defined globally or passed if not)
    # If it's not global, you'll need to define it here or pass it as an argument
    connections = [ 
        (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
        (11, 12), (11, 13), (12, 14), (13, 15), (14, 16), (5, 11), (6, 12)
    ]
    plotted_points = []
    for i in range(len(keypoints)): # Should be 17 keypoints
        # keypoints here are the scaled_kpts, where each element is (nx, ny, v) or (None, None, v)
        x_scaled, y_scaled, conf = keypoints[i] 
        
        if x_scaled is not None and y_scaled is not None and conf > 0.1: # Check if point is valid
            cv2.circle(canvas, (int(x_scaled), int(y_scaled)), thickness + 1, color, -1)
            plotted_points.append((int(x_scaled), int(y_scaled), conf))
        else:
            plotted_points.append((None, None, 0)) # Mark as invalid for drawing connections

    for i, (p1_idx, p2_idx) in enumerate(connections):
        # Ensure indices are within bounds of plotted_points
        if p1_idx < len(plotted_points) and p2_idx < len(plotted_points):
            if plotted_points[p1_idx][0] is not None and plotted_points[p2_idx][0] is not None:
                cv2.line(canvas, plotted_points[p1_idx][:2], plotted_points[p2_idx][:2], color, thickness)



def draw_focused_skeleton_inset(main_frame, raw_keypoints_list_of_lists, position, size, bg_color):
    """Draws the skeleton from raw_keypoints onto a small inset on the main_frame."""
    if not raw_keypoints_list_of_lists or not raw_keypoints_list_of_lists[0]: # Check if it's [[]] or []
        return

    raw_keypoints = raw_keypoints_list_of_lists[0] # Get the actual list of 17 keypoints

    inset_w, inset_h = size
    inset_x, inset_y = position

    # Create a small canvas for the inset
    inset_canvas = np.full((inset_h, inset_w, 3), bg_color, dtype=np.uint8)

    # Draw the skeleton onto the inset canvas
    # Need to normalize keypoints relative to their own bounding box then scale to inset_canvas size
    # For simplicity now, just drawing points if visible
    # A proper visualization would re-scale these keypoints to fit nicely in the inset.
    # This is a placeholder for a more robust skeleton drawing scaled to the inset.
    
    # Crude way to scale: find min/max of detected points to roughly center
    valid_pts = [pt for pt in raw_keypoints if pt[2] > 0.1]
    if len(valid_pts) > 1:
        xs = [pt[0] for pt in valid_pts]
        ys = [pt[1] for pt in valid_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        kpt_w = max_x - min_x
        kpt_h = max_y - min_y

        scale_x = inset_w / kpt_w if kpt_w > 0 else 1
        scale_y = inset_h / kpt_h if kpt_h > 0 else 1
        scale = min(scale_x, scale_y) * 0.8 # 80% of available space

        scaled_kpts = []
        for x,y,v in raw_keypoints:
            if v > 0.1:
                # Translate to origin, scale, then translate to center of inset
                nx = int((x - min_x) * scale + (inset_w - kpt_w * scale) / 2)
                ny = int((y - min_y) * scale + (inset_h - kpt_h * scale) / 2)
                scaled_kpts.append((nx,ny,v))
            else:
                scaled_kpts.append((None,None,v)) # Use None to signify not to draw
        draw_skeleton(inset_canvas, scaled_kpts, color=(0, 255, 255)) # Cyan skeleton
    
    # Overlay the inset onto the main frame
    if inset_y + inset_h <= main_frame.shape[0] and inset_x + inset_w <= main_frame.shape[1]:
        main_frame[inset_y : inset_y + inset_h, inset_x : inset_x + inset_w] = inset_canvas
    cv2.putText(main_frame, "Focus Frame", (inset_x, inset_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)


# --- 6. Video Capture and Processing ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open video source")
    exit()

original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # Renamed from width
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Renamed from height
new_width = int(original_width * DISPLAY_SCALE)
new_height = int(original_height * DISPLAY_SCALE)

keypoint_buffer = deque(maxlen=FRAMES_BUFFER_SIZE)
raw_keypoints_buffer_for_viz = deque(maxlen=FRAMES_BUFFER_SIZE) # NEW: Store raw keypoints for focused frame viz

frame_count = 0
predicted_action = "Initializing..."
last_probabilities = np.zeros(len(ACTION_CLASSES))
last_attention_weights = np.zeros(FRAMES_BUFFER_SIZE)

# --- Matplotlib Setup ---
plt.ion()
fig_prob, ax_prob = plt.subplots(figsize=(10, 6))
plot_frame_numbers = deque(maxlen=FRAMES_BUFFER_SIZE)
plot_probabilities = [deque(maxlen=FRAMES_BUFFER_SIZE) for _ in ACTION_CLASSES]
lines = []
colors_plot = plt.cm.viridis(np.linspace(0, 1, len(ACTION_CLASSES))) # Renamed
for i, action_cls_name in enumerate(ACTION_CLASSES):
    line, = ax_prob.plot([], [], label=action_cls_name, color=colors_plot[i], linewidth=2)
    lines.append(line)
ax_prob.set_xlabel('Frame Number (Buffer Window)', fontsize=12)
ax_prob.set_ylabel('Probability', fontsize=12)
ax_prob.set_title(f'Action Probabilities (Window: {FRAMES_BUFFER_SIZE} frames, Plot Update: {PLOT_UPDATE_INTERVAL} frames)', fontsize=14)
ax_prob.set_ylim([0, 1.1])
ax_prob.legend(loc='upper left')
ax_prob.grid(True)
fig_prob.tight_layout()

# --- Main Loop ---
processing_start_time = time.time()
while True:
    loop_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    results = pose_model(frame, verbose=False, conf=0.5)
    annotated_frame = results[0].plot() # YOLO's default plot

    current_raw_keypoints_for_viz = None # For the focus frame inset
    current_frame_keypoints_for_buffer = [] # For PoseDataset
    try:
        if len(results) > 0 and results[0].keypoints is not None and results[0].keypoints.data.numel() > 0:
            person_keypoints_np = results[0].keypoints.data[0].cpu().numpy() # (17,3) array
            current_raw_keypoints_for_viz = person_keypoints_np.tolist() # Store for focus viz
            current_frame_keypoints_for_buffer = [person_keypoints_np.tolist()] # Expected by PoseDataset
            # print(f"DEBUG: Successfully extracted keypoints for frame {frame_count}")
        else:
            # print(f"DEBUG: No keypoints detected or data empty for frame {frame_count}")
            current_frame_keypoints_for_buffer = [] # Empty list if no detection
            current_raw_keypoints_for_viz = None
    except Exception as e:
        # print(f"Error during keypoint extraction: {e}")
        current_frame_keypoints_for_buffer = []
        current_raw_keypoints_for_viz = None

    keypoint_buffer.append({"frame_num": frame_count, "keypoints": current_frame_keypoints_for_buffer})
    raw_keypoints_buffer_for_viz.append(current_raw_keypoints_for_viz) # Store raw kpts for visualization

    if len(keypoint_buffer) == FRAMES_BUFFER_SIZE:
        keypoints_for_prediction = list(keypoint_buffer)
        
        predicted_action, current_probabilities, current_attention_weights = predict_action(
            action_model,
            keypoints_for_prediction
        )
        last_probabilities = current_probabilities
        last_attention_weights = current_attention_weights

        # Update Matplotlib Probability Plot
        plot_frame_numbers.append(frame_count)
        for i, prob in enumerate(current_probabilities):
            plot_probabilities[i].append(prob)
        if frame_count % PLOT_UPDATE_INTERVAL == 0:
            # ... (Matplotlib plot update code remains the same)
            if len(plot_frame_numbers) > 0 and all(len(p) == len(plot_frame_numbers) for p in plot_probabilities):
                try:
                    for i, line_plt in enumerate(lines): # Renamed line to line_plt
                        line_plt.set_data(list(plot_frame_numbers), list(plot_probabilities[i]))
                    if plot_frame_numbers:
                        ax_prob.set_xlim(plot_frame_numbers[0], plot_frame_numbers[-1] + 1)
                    ax_prob.relim()
                    ax_prob.autoscale_view(scalex=False, scaley=True)
                    fig_prob.canvas.draw_idle()
                    fig_prob.canvas.flush_events()
                except Exception as e:
                    print(f"Error updating probability plot: {e}")


    # Display Results
    cv2.putText(annotated_frame, f"Action: {predicted_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_pos = 60
    for i, action_cls_name in enumerate(ACTION_CLASSES):
        prob_text = f"{action_cls_name}: {last_probabilities[i]:.2f}"
        cv2.putText(annotated_frame, prob_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_pos += 25

    # Draw Attention Weights
    if len(keypoint_buffer) == FRAMES_BUFFER_SIZE and last_attention_weights is not None:
        draw_attention_bars(annotated_frame, last_attention_weights,
                            ATTENTION_VIZ_X_START, ATTENTION_VIZ_Y_START, # Y is bottom
                            ATTENTION_VIZ_BAR_WIDTH_TOTAL, ATTENTION_VIZ_HEIGHT,
                            bar_color=(255, 165, 0)) # Orange

    # Draw Focused Frame Inset
    if len(keypoint_buffer) == FRAMES_BUFFER_SIZE and last_attention_weights is not None and np.any(last_attention_weights): # Check if any weight is non-zero
        focused_frame_buffer_idx = np.argmax(last_attention_weights)
        # The raw_keypoints_buffer_for_viz stores keypoints chronologically,
        # but the attention weights correspond to the input sequence for the model
        # which is padded at the start.
        # If padding in PoseDataset.process_keypoints pads with T_pad frames,
        # then attention_weights[k] corresponds to raw_keypoints_buffer_for_viz[k - T_pad]
        # For simplicity, if using start-padding, attention_weights[k] for k >= T_pad
        # corresponds to an actual frame. The first T_pad attention weights might be for padded zeros.

        # Let's find the index in the *actual* raw keypoints buffer.
        # This assumes that process_keypoints uses data from the end of the buffer
        # and pads at the beginning if the buffer isn't full.
        # If the buffer is full, then the indices align directly.
        if len(raw_keypoints_buffer_for_viz) == FRAMES_BUFFER_SIZE:
            keypoints_of_focused_frame = list(raw_keypoints_buffer_for_viz)[focused_frame_buffer_idx]
            if keypoints_of_focused_frame: # If keypoints were detected for that frame
                 draw_focused_skeleton_inset(annotated_frame, [keypoints_of_focused_frame], # Wrap in list for function
                                            (FOCUSED_FRAME_INSET_X, FOCUSED_FRAME_INSET_Y),
                                            FOCUSED_FRAME_INSET_SIZE,
                                            FOCUSED_FRAME_BG_COLOR)

    loop_time = time.time() - loop_start_time
    current_fps = 1.0 / loop_time if loop_time > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (new_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    try:
        display_frame = cv2.resize(annotated_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imshow("Action Recognition", display_frame)
    except Exception as e:
        print(f"Error resizing/displaying frame: {e}")

    frame_count += 1
    if cv2.waitKey(1) == ord('q'):
        break

# --- Cleanup ---
processing_end_time = time.time()
# ... (cleanup code remains the same) ...
total_time = processing_end_time - processing_start_time
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds (Avg Overall FPS: {avg_fps:.2f})")

cap.release()
cv2.destroyAllWindows()
plt.ioff()
if len(plot_frame_numbers) > 0:
    print("Showing final probability plot. Close the plot window to exit.")
    try:
        for i, line_plt in enumerate(lines):
            line_plt.set_data(list(plot_frame_numbers), list(plot_probabilities[i]))
        if plot_frame_numbers:
            ax_prob.set_xlim(plot_frame_numbers[0], plot_frame_numbers[-1] + 1)
        ax_prob.relim()
        ax_prob.autoscale_view(scalex=False, scaley=True)
        plt.show() 
    except Exception as e:
        print(f"Error preparing final probability plot: {e}")