import cv2
import numpy as np
from ultralytics import YOLO
import torch
import torch.nn as nn
import time
from collections import deque, defaultdict
import threading
import queue
import copy # For deep copying shared data if needed

# --- Configuration ---
POSE_MODEL_PATH = "yolo11n-pose.pt" # Use 'n' or 'm' as needed
ACTION_MODEL_PATH = "./models/Guardian_best_model.pth"
ACTION_CLASSES = ["Falling", "No Action", "Waving"]
FRAMES_BUFFER_SIZE = 35  # Sequence length for action model input
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Action Model Hyperparameters (Must match the trained model)
INPUT_SIZE = 34
HIDDEN_SIZE = 256
NUM_LAYERS = 4
NUM_CLASSES = len(ACTION_CLASSES)
DROPOUT_RATE = 0.5

# Display settings
DISPLAY_SCALE = 0.75

# Threading Queues and Lock
MAX_QUEUE_SIZE = 5 # Max frames to buffer in queue
frame_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
results_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE) # Queue to pass results for display
stop_event = threading.Event() # Event to signal threads to stop
tracked_persons_lock = threading.Lock() # Lock for accessing shared tracked_persons_data

# --- Preprocessing Logic (Keep as before or import) ---
class PosePreprocessor:
    def __init__(self, max_frames=FRAMES_BUFFER_SIZE, feature_size=INPUT_SIZE):
        self.max_frames = max_frames
        self.feature_size = feature_size
        self.alpha = 0.8
        self.conf_threshold = 0.2

    def process_keypoints_sequence(self, keypoints_sequence_for_person):
        all_frames_features = []
        previous_frame_normalized = None
        relevant_frames_data = list(keypoints_sequence_for_person)[-self.max_frames:]

        for frame_data in relevant_frames_data:
            processed_feature_vector = np.zeros(self.feature_size)
            try:
                if frame_data is None or 'keypoints_for_id' not in frame_data or \
                   not isinstance(frame_data['keypoints_for_id'], list) or \
                   len(frame_data['keypoints_for_id']) != 17:
                    all_frames_features.append(processed_feature_vector)
                    previous_frame_normalized = None
                    continue

                frame_keypoints_np = np.array(frame_data['keypoints_for_id']).reshape(-1, 3)
                if frame_keypoints_np.shape != (17, 3):
                    all_frames_features.append(processed_feature_vector)
                    previous_frame_normalized = None
                    continue

                valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > self.conf_threshold]
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
                    normalized_kps[:, :2] = self.alpha * normalized_kps[:, :2] + \
                                            (1 - self.alpha) * previous_frame_normalized[:, :2]
                previous_frame_normalized = normalized_kps
                processed_feature_vector = normalized_kps[:, :2].flatten()
                if processed_feature_vector.shape[0] != self.feature_size:
                    processed_feature_vector = np.zeros(self.feature_size)

            except Exception as e:
                processed_feature_vector = np.zeros(self.feature_size)
                previous_frame_normalized = None
            all_frames_features.append(processed_feature_vector)

        num_processed = len(all_frames_features)
        if num_processed == 0: return None
        padded_features = np.zeros((self.max_frames, self.feature_size))
        start_idx_pad = max(0, self.max_frames - num_processed)
        for i in range(num_processed):
            if all_frames_features[i].shape[0] == self.feature_size:
                padded_features[start_idx_pad + i, :] = all_frames_features[i]
        if padded_features.shape != (self.max_frames, self.feature_size): return None
        return padded_features

# --- Model Definition (Keep as before or import) ---
class AttentionLayer(nn.Module):
    # ...(keep implementation)
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
    def forward(self, lstm_output):
        scores = torch.tanh(self.attention_weights(lstm_output))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        return context_vector, attention_weights.squeeze(-1)

class ActionRecognitionBiLSTMWithAttention(nn.Module):
     # ...(keep implementation)
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

# --- Load Models ---
# Models should ideally be loaded within the thread that uses them if using multiprocessing,
# but for threading, loading them globally might be okay, especially for read-only use.
# Ensure PyTorch/YOLO operations are thread-safe if models are shared.
# Loading them globally here for simplicity with threading.
pose_model = YOLO(POSE_MODEL_PATH)
action_model = ActionRecognitionBiLSTMWithAttention(
    INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES, DROPOUT_RATE
).to(DEVICE)
try:
    action_model.load_state_dict(torch.load(ACTION_MODEL_PATH, map_location=DEVICE))
    print("Action recognition model loaded successfully.")
    action_model.eval()
except Exception as e:
    print(f"Error loading action model: {e}")
    exit()

# --- Global State for Tracked Persons (Accessed via Lock) ---
tracked_persons_data = defaultdict(lambda: {
    'keypoint_buffer': deque(maxlen=FRAMES_BUFFER_SIZE),
    'action': "Initializing...",
    'probabilities': np.zeros(len(ACTION_CLASSES)),
    'last_seen': time.time(),
    'color': (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)),
    'bbox': None # Store current bounding box
})

# --- Preprocessor Instance ---
preprocessor = PosePreprocessor(max_frames=FRAMES_BUFFER_SIZE, feature_size=INPUT_SIZE)

# --- Action Prediction Function (Used by Processing Thread) ---
def predict_person_action(model, person_keypoints_sequence):
    model.eval()
    normalized_features = preprocessor.process_keypoints_sequence(person_keypoints_sequence)
    default_probs = np.zeros(len(ACTION_CLASSES))
    no_action_idx = ACTION_CLASSES.index("No Action") if "No Action" in ACTION_CLASSES else 0
    default_probs[no_action_idx] = 1.0
    default_action = ACTION_CLASSES[no_action_idx]

    if normalized_features is None: return default_action, default_probs
    input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities_tensor, _ = model(input_tensor)
        probabilities = probabilities_tensor.cpu().numpy()[0]
    predicted_index = np.argmax(probabilities)
    predicted_action = ACTION_CLASSES[predicted_index]
    return predicted_action, probabilities

# --- Thread 1: Frame Grabber ---
def frame_grabber(cap, queue, stop_event):
    print("Frame grabber thread started.")
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Frame grabber: Failed to grab frame or stream ended.")
            stop_event.set() # Signal other threads to stop
            break
        if not queue.full():
            queue.put(frame) # Put frame into the queue
        else:
            # Optional: If queue is full, maybe drop the oldest frame and add the new one
            # Or just wait shortly (can cause lag if processing is too slow)
            try:
                queue.get_nowait() # Discard oldest
                queue.put(frame)
            except queue.Empty:
                 pass
            # time.sleep(0.001) # Small sleep to prevent busy-waiting if queue is full
    print("Frame grabber thread stopped.")

# --- Thread 2: Processing (Tracking & Action Recognition) ---
def processing_worker(frame_queue, results_queue, stop_event):
    print("Processing thread started.")
    frame_count = 0 # Local frame count for this thread
    while not stop_event.is_set():
        try:
            frame = frame_queue.get(timeout=1) # Wait up to 1 second for a frame
        except queue.Empty:
            continue # No frame available, continue waiting

        current_time = time.time()

        # 1. Pose Estimation and Tracking
        try:
            results = pose_model.track(frame, persist=True, verbose=False, conf=0.5) # Use tracking
        except Exception as e:
            print(f"Error during pose tracking: {e}")
            continue # Skip frame on error

        # Prepare data structure for results queue
        processed_results = {
            'frame': frame.copy(), # Send a copy of the original frame
            'tracked_info': {} # Store info per ID for this frame
        }

        # 2. Process Each Tracked Person found in THIS frame
        current_frame_ids = []
        if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
            boxes = results[0].boxes.data.cpu().numpy()
            keypoints = results[0].keypoints.data.cpu().numpy() # (N, 17, 3)
            track_ids = results[0].boxes.id.int().cpu().tolist()
            min_len = min(len(boxes), len(keypoints), len(track_ids))

            for i in range(min_len):
                person_id = track_ids[i]
                current_frame_ids.append(person_id)
                person_keypoints = keypoints[i] # Shape (17, 3)
                person_bbox = boxes[i][:4] # x1, y1, x2, y2

                # --- Acquire lock to safely update shared state ---
                with tracked_persons_lock:
                    # Update last seen time and bbox
                    tracked_persons_data[person_id]['last_seen'] = current_time
                    tracked_persons_data[person_id]['bbox'] = person_bbox

                    # Add current keypoints to this person's buffer
                    frame_data_for_buffer = {
                        "frame_num": frame_count,
                        "keypoints_for_id": person_keypoints.tolist()
                    }
                    tracked_persons_data[person_id]['keypoint_buffer'].append(frame_data_for_buffer)

                    # Predict action IF buffer is full
                    if len(tracked_persons_data[person_id]['keypoint_buffer']) == FRAMES_BUFFER_SIZE:
                        buffer_to_predict = tracked_persons_data[person_id]['keypoint_buffer'] # Pass the deque
                        try:
                            # Make a deep copy if predict_person_action modifies it, though it shouldn't
                            pred_action, pred_probs = predict_person_action(action_model, buffer_to_predict)
                            # Store results back in the shared state
                            tracked_persons_data[person_id]['action'] = pred_action
                            tracked_persons_data[person_id]['probabilities'] = pred_probs
                        except Exception as e:
                            print(f"Error during action prediction for ID {person_id}: {e}")
                            # Keep previous action/probs on error

                    # --- End lock ---

        # 3. (Optional) Cleanup Old Tracks (Done outside the loop iterating current detections)
        ids_to_remove = []
        with tracked_persons_lock:
            for pid, data in tracked_persons_data.items():
                 if current_time - data['last_seen'] > 5.0: # Remove if not seen for 5 seconds
                     ids_to_remove.append(pid)
            for pid in ids_to_remove:
                del tracked_persons_data[pid]
            # --- End lock ---
            # print(f"Active track IDs: {list(tracked_persons_data.keys())}") # Debug: Show active IDs

        # 4. Prepare results for display thread (copy necessary info)
        with tracked_persons_lock:
            # Create a deep copy to avoid race conditions when display thread reads it
            processed_results['tracked_info'] = copy.deepcopy(tracked_persons_data)
             # Filter results to only include IDs seen in *this* frame if needed for drawing?
             # No, display needs the latest state for *all* active tracks, even if not in frame
             # However, drawing should only happen for tracks with a valid bbox from this frame
             # We store bbox inside tracked_persons_data now.

        # 5. Put results into the results queue
        if not results_queue.full():
            results_queue.put(processed_results)
        else:
            try:
                results_queue.get_nowait() # Discard oldest result
                results_queue.put(processed_results)
            except queue.Empty:
                 pass

        frame_count += 1

    print("Processing thread stopped.")


# --- Main Thread (Visualization) ---
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("http://172.19.44.170:8080/video") # Your network stream

    if not cap.isOpened():
        print("Cannot open video source")
        exit()

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = int(width * DISPLAY_SCALE)
    new_height = int(height * DISPLAY_SCALE)

    # Start the threads
    grabber = threading.Thread(target=frame_grabber, args=(cap, frame_queue, stop_event), daemon=True)
    processor = threading.Thread(target=processing_worker, args=(frame_queue, results_queue, stop_event), daemon=True)
    grabber.start()
    processor.start()

    processing_start_time = time.time()
    display_frame_count = 0

    while not stop_event.is_set():
        loop_start = time.time()

        try:
            # Get the latest processed results
            results_data = results_queue.get(timeout=0.1) # Wait briefly for results
            frame_to_display = results_data['frame']
            tracked_info_snapshot = results_data['tracked_info'] # Get the state snapshot
        except queue.Empty:
            # No new results, maybe display the last frame or skip?
            # Continue to allow checking stop_event
            time.sleep(0.01) # Prevent busy-waiting
            continue

        # --- Display Annotations ---
        # Draw based on the tracked_info_snapshot
        for person_id, data in tracked_info_snapshot.items():
            bbox = data.get('bbox') # Get bbox stored by processing thread
            if bbox is not None:
                x1, y1, x2, y2 = map(int, bbox[:4])
                color = data['color']
                action = data['action']
                # Draw BBox
                cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), color, 2)
                # Display ID and Action
                label_text = f"ID: {person_id} - {action}"
                (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                # Ensure text background doesn't go off-screen
                text_y = y1 - 10 if y1 - h - 15 > 0 else y1 + h + 5
                bg_y1 = text_y - h - 5
                bg_y2 = text_y + 5
                cv2.rectangle(frame_to_display, (x1, bg_y1), (x1 + w, bg_y2), color, -1)
                cv2.putText(frame_to_display, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


        # Display FPS (calculated based on display loop rate)
        loop_time = time.time() - loop_start
        current_fps = 1.0 / loop_time if loop_time > 0 else 0
        cv2.putText(frame_to_display, f"Display FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        try:
            display_frame_resized = cv2.resize(frame_to_display, (new_width, new_height), interpolation=cv2.INTER_AREA)
            cv2.imshow("Multi-Person Action Recognition (Threaded)", display_frame_resized)
            display_frame_count += 1
        except Exception as e:
            print(f"Error resizing/displaying frame: {e}")

        # Check for exit key
        if cv2.waitKey(1) == ord('q'):
            stop_event.set() # Signal threads to stop
            break

    # --- Cleanup ---
    print("Stopping threads...")
    stop_event.set()
    grabber.join(timeout=2)
    processor.join(timeout=5) # Give processor more time to finish current frame
    cap.release()
    cv2.destroyAllWindows()

    processing_end_time = time.time()
    total_time = processing_end_time - processing_start_time
    # Note: display_frame_count is frames *shown*, not processed
    avg_display_fps = display_frame_count / total_time if total_time > 0 else 0
    print(f"\nDisplayed {display_frame_count} frames in {total_time:.2f} seconds (Avg Display FPS: {avg_display_fps:.2f})")
    print("Exiting.")