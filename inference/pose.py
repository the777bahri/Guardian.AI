import numpy as np

class PosePreprocessor:
    def __init__(self, max_frames, feature_size):
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