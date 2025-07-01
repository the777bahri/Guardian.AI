import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
# import cv2 # Removed duplicate import
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
from collections import deque

# --- OPTIMIZATION ---
# Update plot only every N frames to reduce rendering overhead
plot_update_interval = 10 # Update plot every 10 frames (adjust as needed)

# --- 1. Data Preprocessing (PoseDataset) ---
class PoseDataset(object):
    def __init__(self, data_dir, action_classes, max_frames=35):
        self.data_dir = data_dir
        self.action_classes = action_classes
        self.max_frames = max_frames
        self.feature_size = 34 # Expected feature size (17 keypoints * 2 coords)

    def process_keypoints(self, keypoints_data):
        all_frames_keypoints = []
        previous_frame_normalized = None # Store the *normalized* previous frame for EMA
        alpha = 0.8 # Smoothing factor for EMA
        conf_threshold = 0.2

        # Ensure we only process up to max_frames from the end
        relevant_keypoints_data = keypoints_data[-self.max_frames:]

        for frame_data in relevant_keypoints_data:
            processed_frame = np.zeros(self.feature_size) # Default to zeros for this frame
            try:
                if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                    # print(f"Skipping invalid frame data structure: {frame_data}")
                    all_frames_keypoints.append(processed_frame) # Append zeros
                    continue

                frame_keypoints = frame_data['keypoints']
                # Check if keypoints list is valid and contains data for at least one person
                if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0 or \
                   not isinstance(frame_keypoints[0], list) or len(frame_keypoints[0]) == 0:
                    # print("Frame keypoints list is invalid or empty for first person")
                    all_frames_keypoints.append(processed_frame) # Append zeros
                    continue

                # Convert first person's keypoints, ensure shape (17, 3)
                frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    # print(f"Incorrect keypoint shape: {frame_keypoints_np.shape}")
                    all_frames_keypoints.append(processed_frame) # Append zeros
                    continue

                # Filter out keypoints with low confidence for normalization stats
                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > conf_threshold]
                if valid_keypoints.shape[0] < 2: # Need at least 2 points for mean/std
                    # print("Not enough valid keypoints for normalization")
                    all_frames_keypoints.append(processed_frame) # Append zeros
                    continue

                # Z-Score Normalization for X and Y based on valid points
                mean_x = np.mean(valid_keypoints[:, 0])
                std_x = np.std(valid_keypoints[:, 0]) + 1e-8 # Add epsilon
                mean_y = np.mean(valid_keypoints[:, 1])
                std_y = np.std(valid_keypoints[:, 1]) + 1e-8 # Add epsilon

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
                     # print(f"Warning: Flattened frame size mismatch {processed_frame.shape}, using zeros.")
                     processed_frame = np.zeros(self.feature_size) # Fallback to zeros

            except Exception as e:
                # print(f"Error processing frame keypoints: {e}. Appending zeros.")
                processed_frame = np.zeros(self.feature_size) # Append zeros on error

            all_frames_keypoints.append(processed_frame)


        # Padding (Pad at the BEGINNING)
        num_processed = len(all_frames_keypoints)
        if num_processed == 0:
             # print("No frames were processed successfully.")
             return None # Indicate failure if absolutely no frames worked

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
            # print(f"FATAL: Final padded shape incorrect: {padded_keypoints.shape}")
            # This indicates a major logic error if it occurs
            return None # Or handle error appropriately

        return padded_keypoints


# --- 2. Model Definition (ActionRecognitionBiLSTMWithAttention) ---
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1) # hidden_size * 2 for BiLSTM

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size * 2)
        scores = torch.tanh(self.attention_weights(lstm_output)) # Apply tanh for better gradient flow (optional)
        # scores = self.attention_weights(lstm_output)  # (batch_size, seq_length, 1)
        attention_weights = torch.softmax(scores, dim=1) # Normalize scores to weights
        # Context vector: weighted sum of LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_output, dim=1) # (batch_size, hidden_size * 2)
        return context_vector, attention_weights.squeeze(-1) # Return context and weights (batch_size, seq_length)

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
        self.dropout = nn.Dropout(dropout_rate) # Apply dropout before the final layer

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
        context_vector, attention_weights = self.attention(lstm_out) # context_vector: (batch_size, hidden_size * 2)

        # Final classification layer
        # Apply dropout to context vector before FC layer
        context_vector_dropped = self.dropout(context_vector)
        logits = self.fc(context_vector_dropped) # logits: (batch_size, num_classes)

        # Apply Softmax to get probabilities - Model now outputs probabilities
        probabilities = torch.softmax(logits, dim=1)

        # Return probabilities and attention weights
        return probabilities, attention_weights


# --- 3. Configuration ---
pose_model_path = "yolo11n-pose.pt" # Assuming 'yolo11n-pose.pt' was a typo? Use standard name.
action_model_path = "models/Guardian_best_model.pth"
# action_model_path = "training/BiLSTMWithAttention_best_model.pth"
action_classes = ["Falling", "No Action", "Waving"]
frames_per_clip = 35 # Buffer size and sequence length for processing
lstm_input_frames = frames_per_clip # Ensure consistency
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 4. Load Models ---
pose_model = YOLO(pose_model_path)
input_size = 34 # 17 keypoints * 2 (x and y)
hidden_size = 256
num_layers = 4
num_classes = len(action_classes)
dropout_rate = 0.5

action_model = ActionRecognitionBiLSTMWithAttention(input_size, hidden_size, num_layers, num_classes, dropout_rate).to(device)

try:
    action_model.load_state_dict(torch.load(action_model_path, map_location=device))
    print("Action recognition model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Action model file not found at {action_model_path}")
    exit()
except Exception as e:
    print(f"Error loading action model state_dict: {e}")
    # Consider printing more details or raising the exception for debugging
    # raise e
    exit()

action_model.eval() # Set model to evaluation mode

# --- 5. Prediction Function ---
def predict_action(model, keypoints_sequence, action_classes, device, sequence_length):
    model.eval() # Ensure model is in eval mode
    # Instantiate PoseDataset with the correct sequence length for processing
    pose_dataset = PoseDataset("data", action_classes, max_frames=sequence_length)
    normalized_keypoints = pose_dataset.process_keypoints(keypoints_sequence)

    default_probs = np.zeros(len(action_classes))
    no_action_idx = action_classes.index("No Action") if "No Action" in action_classes else 0
    default_probs[no_action_idx] = 1.0

    if normalized_keypoints is None:
        # print("Prediction skipped: Processing failed.")
        return action_classes[no_action_idx], default_probs

    # Shape check (already done in process_keypoints padding, but good to double-check)
    if normalized_keypoints.shape != (sequence_length, input_size):
         # print(f"Warning: Final keypoints shape mismatch. Expected {(sequence_length, input_size)}, Got {normalized_keypoints.shape}. Skipping.")
         return action_classes[no_action_idx], default_probs

    # Add batch dimension and send to device
    normalized_keypoints_tensor = torch.tensor(normalized_keypoints, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        # Model now returns (probabilities, attention_weights)
        probabilities_tensor, _ = model(normalized_keypoints_tensor)

        # Get probabilities for the single batch item
        probabilities = probabilities_tensor.cpu().numpy()[0] # Shape (num_classes,)

        # Ensure probabilities sum to 1 (optional check)
        # if not np.isclose(np.sum(probabilities), 1.0):
        #     print(f"Warning: Probabilities do not sum to 1: {probabilities}")

        predicted_index = np.argmax(probabilities)
        predicted_action = action_classes[predicted_index]

    return predicted_action, probabilities


# --- 6. Video Capture and Processing ---

# --- Suggestion: Test with a local file first if network stream is laggy ---
# cap = cv2.VideoCapture("./FallDetection/falldown.mp4") # Example local file
# cap = cv2.VideoCapture("http://172.19.44.237:8080/video") # Your network stream
cap = cv2.VideoCapture(0) # Webcam

if not cap.isOpened():
    print("Cannot open video source")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
display_scale = 1.00
new_width = int(width * display_scale)
new_height = int(height * display_scale)

keypoint_buffer = deque(maxlen=lstm_input_frames)
frame_count = 0
predicted_action = "Initializing..."
last_probabilities = np.zeros(len(action_classes))

# --- Matplotlib Setup ---
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
plot_frame_numbers = deque(maxlen=lstm_input_frames)
plot_probabilities = [deque(maxlen=lstm_input_frames) for _ in action_classes]
lines = []
colors = plt.cm.viridis(np.linspace(0, 1, len(action_classes)))
for i, action in enumerate(action_classes):
    line, = ax.plot([], [], label=action, color=colors[i], linewidth=2)
    lines.append(line)
ax.set_xlabel('Frame Number (within buffer)', fontsize=12)
ax.set_ylabel('Probability', fontsize=12)
# Updated title to reflect plot update interval
ax.set_title(f'Action Probabilities (Window: {lstm_input_frames} frames, Plot Update: {plot_update_interval} frames)', fontsize=14)
ax.set_ylim([0, 1.1])
ax.legend(loc='upper left')
ax.grid(True)
fig.tight_layout() # Adjust layout

# --- Main Loop ---
processing_start_time = time.time()

while True:
    loop_start = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 1. Pose Estimation
    results = pose_model(frame, verbose=False, conf=0.5)
    # Use a copy for annotation to avoid modifying the original frame if needed elsewhere
    annotated_frame = results[0].plot()

    # Extract keypoints with added robustness
    current_frame_keypoints = [] # Default to empty
    try:
        if len(results) > 0 and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            # Convert tensor to list [[x1,y1,c1], [x2,y2,c2], ...]
            person_keypoints_list = results[0].keypoints.data[0].cpu().numpy().tolist()
            # YOLO output might be slightly different; ensure it's a list of lists/tuples
            if isinstance(person_keypoints_list, list) and len(person_keypoints_list) == 17:
                 # Wrap in another list to match the expected structure [[kpt1], [kpt2]...]
                 # This structure `[[[x,y,c],...]]` seems more likely from debug? Let's keep original logic for now.
                 # Check original extraction logic again if processing fails.
                 # Assuming processing expects format: [{"frame_num": n, "keypoints": [[[x,y,c], ...]]}]
                  current_frame_keypoints = results[0].keypoints.data.cpu().numpy().tolist() # Original logic was likely correct
                  if len(current_frame_keypoints) > 0 and not isinstance(current_frame_keypoints[0], list):
                        current_frame_keypoints = [current_frame_keypoints] # Wrap if needed [[kpt list]]

    except Exception as e:
        # print(f"Error during keypoint extraction: {e}")
        current_frame_keypoints = [] # Ensure it's empty on error


    keypoint_buffer.append({"frame_num": frame_count, "keypoints": current_frame_keypoints})

    # 2. Action Prediction (only if buffer is full)
    if len(keypoint_buffer) == lstm_input_frames:
        keypoints_for_prediction = list(keypoint_buffer)
        predicted_action, current_probabilities = predict_action(
            action_model,
            keypoints_for_prediction,
            action_classes,
            device,
            lstm_input_frames
        )
        last_probabilities = current_probabilities

        # --- Append Plot Data (always append when prediction happens) ---
        plot_frame_numbers.append(frame_count)
        for i, prob in enumerate(current_probabilities):
            plot_probabilities[i].append(prob)

        # --- OPTIMIZATION: Update Plot Lines Less Frequently ---
        if frame_count % plot_update_interval == 0:
            valid_plot_data = len(plot_frame_numbers) > 0 and all(len(p) == len(plot_frame_numbers) for p in plot_probabilities)
            if valid_plot_data:
                try:
                    plot_render_start = time.time()
                    for i, line in enumerate(lines):
                        # Convert deques to lists for plotting
                        line.set_data(list(plot_frame_numbers), list(plot_probabilities[i]))
                    if plot_frame_numbers:
                        ax.set_xlim(plot_frame_numbers[0], plot_frame_numbers[-1] + 1)
                    # Only autoscale Y if needed, keep X limits fixed relative to buffer
                    ax.relim() # Recalculate Y limits based on visible data
                    ax.autoscale_view(scalex=False, scaley=True)
                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()
                    # print(f"Plot render time: {time.time() - plot_render_start:.4f}")
                except Exception as e:
                    print(f"Error updating plot: {e}") # Catch potential plot errors


    # 3. Display Results
    # Add FPS counter
    loop_time = time.time() - loop_start
    current_fps = 1.0 / loop_time if loop_time > 0 else 0
    cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", (new_width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(annotated_frame, f"Action: {predicted_action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    y_pos = 60
    for i, action in enumerate(action_classes):
        prob_text = f"{action}: {last_probabilities[i]:.2f}"
        cv2.putText(annotated_frame, prob_text, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        y_pos += 25

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
total_time = processing_end_time - processing_start_time
avg_fps = frame_count / total_time if total_time > 0 else 0
print(f"\nProcessed {frame_count} frames in {total_time:.2f} seconds (Avg Overall FPS: {avg_fps:.2f})")

cap.release()
cv2.destroyAllWindows()
plt.ioff()
print("Showing final plot. Close the plot window to exit.")
# Ensure plot is drawn correctly before showing
if len(plot_frame_numbers) > 0:
     try:
        for i, line in enumerate(lines):
            line.set_data(list(plot_frame_numbers), list(plot_probabilities[i]))
        if plot_frame_numbers:
            ax.set_xlim(plot_frame_numbers[0], plot_frame_numbers[-1] + 1)
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
     except Exception as e:
        print(f"Error preparing final plot: {e}")
plt.show()