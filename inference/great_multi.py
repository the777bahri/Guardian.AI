import cv2
import numpy as np
from ultralytics import YOLO
import torch
import time
from collections import deque, defaultdict
import threading
import queue
import copy
import os
from model import ActionRecognitionBiLSTMWithAttention
from pose import PosePreprocessor

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class MultiPersonActionRecognitionApp:
    def __init__(self):
        # --- Configuration ---
        self.TRACK_CFG = "bytetrack.yaml"
        self.POSE_MODEL_PATH = "yolo11n-pose.pt"
        self.ACTION_MODEL_PATH = "training/BiLSTMWithAttention_best_model_3.0.pth"
        self.ACTION_CLASSES = ["Falling", "No Action", "Waving"]
        self.FRAMES_BUFFER_SIZE = 35
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.DEVICE}")

        self.INPUT_SIZE = 34
        self.HIDDEN_SIZE = 256
        self.NUM_LAYERS = 4
        self.NUM_CLASSES = len(self.ACTION_CLASSES)
        self.DROPOUT_RATE = 0.5
        self.DISPLAY_SCALE = 0.75
        self.MAX_QUEUE_SIZE = 5

        self.frame_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.results_queue = queue.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self.stop_event = threading.Event()
        self.tracked_persons_lock = threading.Lock()

        self.pose_model = YOLO(self.POSE_MODEL_PATH)
        self.action_model = ActionRecognitionBiLSTMWithAttention(
            self.INPUT_SIZE, self.HIDDEN_SIZE, self.NUM_LAYERS, self.NUM_CLASSES, self.DROPOUT_RATE
        ).to(self.DEVICE)
        try:
            self.action_model.load_state_dict(torch.load(self.ACTION_MODEL_PATH, map_location=self.DEVICE))
            print("Action recognition model loaded successfully.")
            self.action_model.eval()
        except Exception as e:
            print(f"Error loading action model: {e}")
            exit()

        self.tracked_persons_data = defaultdict(lambda: {
            'keypoint_buffer': deque(maxlen=self.FRAMES_BUFFER_SIZE),
            'action': "Initializing...",
            'probabilities': np.zeros(len(self.ACTION_CLASSES)),
            'last_seen': time.time(),
            'color': (np.random.randint(50, 200), np.random.randint(50, 200), np.random.randint(50, 200)),
            'bbox': None
        })

        self.preprocessor = PosePreprocessor(max_frames=self.FRAMES_BUFFER_SIZE, feature_size=self.INPUT_SIZE)
        self.roi_mask = None
        self.new_width = None
        self.new_height = None
        self.enable_roi = False  # Set to True if you want ROI selection


    def predict_person_action(self, model, person_keypoints_sequence):
        model.eval()
        normalized_features = self.preprocessor.process_keypoints_sequence(person_keypoints_sequence)
        default_probs = np.zeros(len(self.ACTION_CLASSES))
        no_action_idx = self.ACTION_CLASSES.index("No Action") if "No Action" in self.ACTION_CLASSES else 0
        default_probs[no_action_idx] = 1.0
        default_action = self.ACTION_CLASSES[no_action_idx]

        if normalized_features is None:
            return default_action, default_probs
        input_tensor = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            probabilities_tensor, _ = model(input_tensor)
            probabilities = probabilities_tensor.cpu().numpy()[0]
        predicted_index = np.argmax(probabilities)
        predicted_action = self.ACTION_CLASSES[predicted_index]
        return predicted_action, probabilities

    def frame_grabber(self, cap):
        print("Frame grabber thread started.")
        backoff = 0.2
        while not self.stop_event.is_set():
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(backoff)
                continue
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
        print("Frame grabber thread stopped.")

    def processing_worker(self):
        print("Processing thread started.")
        frame_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue
            current_time = time.time()
            try:
                masked_frame = cv2.bitwise_and(frame, frame, mask=self.roi_mask)
                results = self.pose_model.track(
                    masked_frame,
                    persist=True,
                    conf=0.25,
                    iou=0.20,
                    tracker=self.TRACK_CFG,
                    verbose=False)
            except Exception as e:
                print(f"Error during pose tracking: {e}")
                continue

            processed_results = {
                'frame': frame.copy(),
                'tracked_info': {}
            }

            current_frame_ids = []
            if results[0].boxes is not None and results[0].boxes.id is not None and results[0].keypoints is not None:
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
                        frame_data_for_buffer = {
                            "frame_num": frame_count,
                            "keypoints_for_id": person_keypoints.tolist()
                        }
                        self.tracked_persons_data[person_id]['keypoint_buffer'].append(frame_data_for_buffer)
                        if len(self.tracked_persons_data[person_id]['keypoint_buffer']) == self.FRAMES_BUFFER_SIZE:
                            buffer_to_predict = self.tracked_persons_data[person_id]['keypoint_buffer']
                            try:
                                pred_action, pred_probs = self.predict_person_action(self.action_model, buffer_to_predict)
                                self.tracked_persons_data[person_id]['action'] = pred_action
                                self.tracked_persons_data[person_id]['probabilities'] = pred_probs
                            except Exception as e:
                                print(f"Error during action prediction for ID {person_id}: {e}")

            ids_to_remove = []
            with self.tracked_persons_lock:
                for pid, data in self.tracked_persons_data.items():
                    if current_time - data['last_seen'] > 5.0:
                        ids_to_remove.append(pid)
                for pid in ids_to_remove:
                    del self.tracked_persons_data[pid]

            with self.tracked_persons_lock:
                processed_results['tracked_info'] = copy.deepcopy(self.tracked_persons_data)

            if not self.results_queue.full():
                self.results_queue.put(processed_results)
            else:
                try:
                    self.results_queue.get_nowait()
                    self.results_queue.put(processed_results)
                except queue.Empty:
                    pass

            frame_count += 1
        print("Processing thread stopped.")

    def select_roi(self, cap):
        ret, roi_frame = cap.read()
        if not ret:
            print("Failed to read frame for ROI selection.")
            cap.release()
            exit()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.new_width = int(width * self.DISPLAY_SCALE)
        self.new_height = int(height * self.DISPLAY_SCALE)
        roi_display_frame = cv2.resize(roi_frame, (self.new_width, self.new_height))
        roi_points = []
        selection_done = False

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN:
                if roi_points:
                    roi_points.pop()

        roi_window_name = "Define ROI (Left-click: add, Right-click: remove, Enter: confirm)"
        cv2.namedWindow(roi_window_name)
        cv2.setMouseCallback(roi_window_name, mouse_callback)

        while not selection_done:
            if cv2.getWindowProperty(roi_window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("ROI window closed manually. Exiting.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
            temp_frame = roi_display_frame.copy()
            if roi_points:
                for point in roi_points:
                    cv2.circle(temp_frame, point, 5, (0, 255, 0), -1)
                for i in range(1, len(roi_points)):
                    cv2.line(temp_frame, roi_points[i-1], roi_points[i], (255, 0, 0), 2)
            cv2.imshow(roi_window_name, temp_frame)
            key = cv2.waitKey(1)
            if key == 13:
                if len(roi_points) >= 3:
                    selection_done = True
                else:
                    print("Need at least 3 points to define a polygon.")
            elif key == 27:
                print("ROI selection cancelled.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        cv2.destroyWindow(roi_window_name)
        scaled_roi_points = [(int(pt[0] / self.DISPLAY_SCALE), int(pt[1] / self.DISPLAY_SCALE)) for pt in roi_points]
        self.roi_mask = np.zeros(roi_frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(self.roi_mask, [np.array(scaled_roi_points, dtype=np.int32)], 255)
        print("ROI defined. Masking non-ROI areas.")

    def run(self, Source_camera):
        cap = cv2.VideoCapture(int(Source_camera) if str(Source_camera).isdigit() else Source_camera)
        if not cap.isOpened():
            print("Cannot open video source")
            exit()

        if self.enable_roi:
            self.select_roi(cap)
        else:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.new_width = int(width * self.DISPLAY_SCALE)
            self.new_height = int(height * self.DISPLAY_SCALE)
            self.roi_mask = np.ones((height, width), dtype=np.uint8) * 255  # full frame allowed
            print("ROI selection skipped. Using full frame.")

        grabber = threading.Thread(target=self.frame_grabber, args=(cap,), daemon=True)
        processor = threading.Thread(target=self.processing_worker, daemon=True)
        grabber.start()
        processor.start()
        processing_start_time = time.time()
        display_frame_count = 0

        while not self.stop_event.is_set():
            loop_start = time.time()
            try:
                results_data = self.results_queue.get(timeout=0.1)
                frame_to_display = results_data['frame']
                tracked_info_snapshot = results_data['tracked_info']
            except queue.Empty:
                time.sleep(0.01)
                continue
            for person_id, data in tracked_info_snapshot.items():
                bbox = data.get('bbox')
                if bbox is not None:
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    color = data['color']
                    action = data['action']
                    cv2.rectangle(frame_to_display, (x1, y1), (x2, y2), color, 2)
                    label_text = f"ID: {person_id} - {action}"
                    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    text_y = y1 - 10 if y1 - h - 15 > 0 else y1 + h + 5
                    bg_y1 = text_y - h - 5
                    bg_y2 = text_y + 5
                    cv2.rectangle(frame_to_display, (x1, bg_y1), (x1 + w, bg_y2), color, -1)
                    cv2.putText(frame_to_display, label_text, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            loop_time = time.time() - loop_start
            current_fps = 1.0 / loop_time if loop_time > 0 else 0
            cv2.putText(frame_to_display, f"Display FPS: {current_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            try:
                display_frame_resized = cv2.resize(frame_to_display, (self.new_width, self.new_height), interpolation=cv2.INTER_AREA)
                cv2.imshow("Multi-Person Action Recognition (Threaded)", display_frame_resized)
                display_frame_count += 1
            except Exception as e:
                print(f"Error resizing/displaying frame: {e}")
            if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty("Multi-Person Action Recognition (Threaded)", cv2.WND_PROP_VISIBLE) < 1:
                self.stop_event.set()
                break
        print("Stopping threads...")
        self.stop_event.set()
        grabber.join(timeout=2)
        processor.join(timeout=5)
        cap.release()
        cv2.destroyAllWindows()
        processing_end_time = time.time()
        total_time = processing_end_time - processing_start_time
        avg_display_fps = display_frame_count / total_time if total_time > 0 else 0
        print(f"\nDisplayed {display_frame_count} frames in {total_time:.2f} seconds (Avg Display FPS: {avg_display_fps:.2f})")
        print("Exiting.")

if __name__ == "__main__":
    print("Welcome to Multi-Person Action Recognition App")

    # Prompt for video source
    user_input_source = input("Enter video source (e.g., 0 for webcam or a URL like http://...): ").strip()
    # Convert to int if it's a digit (e.g., "0" becomes 0)
    Source_camera = int(user_input_source) if user_input_source.isdigit() else user_input_source

    # Prompt for ROI selection
    user_input_roi = input("Do you want to enable ROI selection? (yes/no): ").strip().lower()
    enable_roi = user_input_roi in ['yes', 'y']

    # Initialize and run app
    app = MultiPersonActionRecognitionApp()
    app.enable_roi = enable_roi
    app.run(Source_camera)
