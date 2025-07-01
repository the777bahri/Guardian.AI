import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset

# Custom dataset for pose data
# This dataset assumes that the data is organized in a directory structure where each action class has its own folder
class PoseDataset(Dataset):
    def __init__(self, data_dir, action_classes, max_frames=40):
        self.data_dir = data_dir
        self.action_classes = action_classes
        self.max_frames = max_frames  # Maximum number of frames per clip
        self.data, self.labels = self.load_data()


    def load_data(self):
        data = []
        labels = []
        for i, action in enumerate(self.action_classes):
            action_dir = os.path.join(self.data_dir, action)
            if not os.path.exists(action_dir):
              print(f"Warning: Directory not found: {action_dir}")  # Debugging
              continue

            for filename in os.listdir(action_dir):
                if filename.endswith("_keypoints.json"):
                    filepath = os.path.join(action_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            keypoints_data = json.load(f)
                            # Extract keypoints and normalize

                            normalized_keypoints = self.process_keypoints(keypoints_data)
                            if normalized_keypoints is not None:
                                data.append(normalized_keypoints)
                                labels.append(i)  # Use index as label
                    except (json.JSONDecodeError, FileNotFoundError) as e:
                        print(f"Error loading or processing {filepath}: {e}")
                        continue  # Skip to the next file

        return data, labels

    def process_keypoints(self, keypoints_data):
        all_frames_keypoints = []
        previous_frame = None  # For temporal smoothing
        alpha = 0.8  # Smoothing factor for EMA

        for frame_data in keypoints_data:
            if not isinstance(frame_data, dict) or 'keypoints' not in frame_data:
                print(f"Skipping invalid frame data: {frame_data}")  # Debugging
                continue  # Skip malformed data

            frame_keypoints = frame_data['keypoints']
            if not isinstance(frame_keypoints, list) or len(frame_keypoints) == 0:
                print("frame keypoints is not a list or is empty")
                continue

            frame_keypoints_np = np.array(frame_keypoints[0]).reshape(-1, 3)  # First person, (17, 3)
            if frame_keypoints_np.shape != (17, 3):
                print(f"Incorrect shape: {frame_keypoints_np.shape}")
                continue

            # Filter out keypoints with low confidence
            valid_keypoints = frame_keypoints_np[frame_keypoints_np[:, 2] > 0.2]
            if valid_keypoints.size == 0:
                continue

            # Z-Score Normalization
            mean_x = np.mean(valid_keypoints[:, 0])
            std_x = np.std(valid_keypoints[:, 0]) + 1e-8  # Avoid division by zero
            mean_y = np.mean(valid_keypoints[:, 1])
            std_y = np.std(valid_keypoints[:, 1]) + 1e-8

            normalized_frame_keypoints = frame_keypoints_np.copy()
            normalized_frame_keypoints[:, 0] = (normalized_frame_keypoints[:, 0] - mean_x) / std_x
            normalized_frame_keypoints[:, 1] = (normalized_frame_keypoints[:, 1] - mean_y) / std_y

            # Temporal Smoothing using EMA
            if previous_frame is not None:
                normalized_frame_keypoints[:, 0] = alpha * normalized_frame_keypoints[:, 0] + (1 - alpha) * previous_frame[:, 0]
                normalized_frame_keypoints[:, 1] = alpha * normalized_frame_keypoints[:, 1] + (1 - alpha) * previous_frame[:, 1]

            previous_frame = normalized_frame_keypoints  # Update for the next iteration

            # Flatten and remove confidence scores
            normalized_frame_keypoints = normalized_frame_keypoints[:, :2].flatten()
            all_frames_keypoints.append(normalized_frame_keypoints)

        # Padding (or truncating)
        if not all_frames_keypoints:
            return None
        padded_keypoints = np.zeros((self.max_frames, all_frames_keypoints[0].shape[0]))
        for i, frame_kps in enumerate(all_frames_keypoints):
            if i < self.max_frames:
                padded_keypoints[i, :] = frame_kps

        return padded_keypoints

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)
