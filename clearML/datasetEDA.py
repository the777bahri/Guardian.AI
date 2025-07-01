import os
import matplotlib
matplotlib.use('Agg') # Set backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import cv2
import numpy as np
import json
from ultralytics import YOLO
# import torch
from ultralytics.utils.plotting import Annotator
from clearml import Task, Dataset
import logging
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')

# --- Helper Functions (count_json_files, plot_json_counts - unchanged) ---
def count_json_files(root_dir):
    # ... (same as before) ...
    json_count = 0
    if not os.path.isdir(root_dir):
        logging.warning(f"Directory not found for counting: {root_dir}")
        return 0
    for _, _, files in os.walk(root_dir):
        json_count += sum(1 for file in files if file.endswith('.json'))
    return json_count

def plot_json_counts(root_dirs, class_labels):
    # ... (same as before) ...
    logging.info(f"Plotting JSON counts for labels: {class_labels}")
    json_counts = []
    valid_labels = []
    for i, rd in enumerate(root_dirs):
         if os.path.isdir(rd):
             count = count_json_files(rd)
             json_counts.append(count)
             valid_labels.append(class_labels[i])
             logging.info(f" - {class_labels[i]}: {count} JSON files")
         else:
             logging.warning(f"Skipping non-existent directory for plotting: {rd}")

    if not json_counts:
         logging.error("No valid directories found to plot JSON counts.")
         return None

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(valid_labels, json_counts, color='skyblue')
    ax.set_xlabel('Action Class')
    ax.set_ylabel('Number of JSON Files')
    ax.set_title('Number of JSON Files per Action Class')
    for i, count in enumerate(json_counts):
         offset = max(json_counts) * 0.01 if max(json_counts) > 0 else 1
         ax.text(i, count + offset, str(count), ha='center', va='bottom')
    plt.tight_layout()
    return fig


# --- Video Generation Function (Switching back to MP4) ---

def keypoints_folder_to_video(folder_path, output_video_path, fps=20.0, yolo_model_path="yolo11n-pose.pt"):
    """
    Converts keypoints from *_keypoints.json files to an MP4 video.
    Returns:
        bool: True on success, False on failure.
    """
    logging.info(f"Attempting video generation for: {folder_path}")
    if not os.path.exists(folder_path):
        logging.error(f"Video Gen Error: Source folder not found: {folder_path}")
        return False

    json_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_keypoints.json')])
    if not json_files:
        logging.warning(f"No '*_keypoints.json' files found in {folder_path}. Cannot generate video.")
        return False

    action_class = os.path.basename(folder_path)
    logging.info(f"Processing class '{action_class}' with {len(json_files)} JSON files.")

    max_width = 0
    max_height = 0
    processed_frames_count = 0

    # --- First pass: Determine frame size ---
    logging.info("First pass: Determining bounding box for keypoints...")
    # (Logic remains the same)
    for i, json_file in enumerate(json_files):
        filepath = os.path.join(folder_path, json_file)
        try:
             with open(filepath, 'r') as f: keypoint_data = json.load(f)
             for frame_data in keypoint_data:
                 if 'keypoints' in frame_data and frame_data['keypoints'] and len(frame_data['keypoints']) > 0:
                     kpts_list = frame_data['keypoints'][0]
                     if isinstance(kpts_list, list) and len(kpts_list) > 0:
                         frame_keypoints = np.array(kpts_list).reshape(-1, 3)
                         valid_keypoints = frame_keypoints[frame_keypoints[:, 2] > 0.2]
                         if valid_keypoints.size > 0:
                             processed_frames_count += 1
                             min_x, max_x = np.min(valid_keypoints[:, 0]), np.max(valid_keypoints[:, 0])
                             min_y, max_y = np.min(valid_keypoints[:, 1]), np.max(valid_keypoints[:, 1])
                             max_width = max(max_width, max_x - min_x)
                             max_height = max(max_height, max_y - min_y)
        except Exception as e:
            logging.warning(f"Video Gen Warning: Error during bounds calc from {filepath}: {e}", exc_info=False)
            continue

    if processed_frames_count == 0 or max_width <= 0 or max_height <= 0:
         logging.error(f"Video Gen Error: No valid keypoints found or max bounds invalid for {action_class}.")
         return False
    logging.info(f"Max keypoint spread: width={max_width:.2f}, height={max_height:.2f}. Frames processed: {processed_frames_count}")

    # --- Setup Video Writer (Using MP4V / .mp4) ---
    padding = 50
    frame_w = int(max_width + 2 * padding)
    frame_h = int(max_height + 2 * padding)
    frame_size = (max(frame_w, 100), max(frame_h, 100)) # W, H format

    # *** CHANGE: Use 'mp4v' fourcc ***
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # The output_video_path will be set to .mp4 in perform_eda

    logging.info(f"Attempting to open VideoWriter: path='{output_video_path}', fourcc='mp4v', fps={fps}, frame_size={frame_size}")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    if not out.isOpened():
        logging.error(f"Video Gen Error: FAILED to open VideoWriter with mp4v codec.")
        # Provide more context if possible
        logging.error("Check if OpenCV has FFMPEG or other suitable backend support installed.")
        return False
    logging.info(f"VideoWriter opened successfully.")

    # --- Load YOLO Model ---
    # (Logic remains the same)
    pose_model = None
    try:
        logging.info(f"Loading YOLO pose model: {yolo_model_path}")
        pose_model = YOLO(yolo_model_path)
        _ = pose_model.predict(np.zeros((64, 64, 3), dtype=np.uint8), verbose=False)
        logging.info("YOLO model loaded.")
    except Exception as e:
        logging.error(f"Video Gen Error: Failed to load YOLO model '{yolo_model_path}'. Annotation may be skipped.", exc_info=True)


    # --- Second pass: Create video frames ---
    # (Logic remains the same - keypoint processing, annotation, text overlay)
    logging.info("Second pass: Generating video frames...")
    frame_write_counter = 0
    json_file_counter = 0
    frames_processed_in_pass2 = 0

    for json_file in json_files:
        filepath = os.path.join(folder_path, json_file)
        json_file_counter += 1
        try:
            with open(filepath, 'r') as f: keypoint_data = json.load(f)
            for frame_data in keypoint_data:
                frames_processed_in_pass2 += 1
                frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
                local_frame_number = frame_data.get('frame', frames_processed_in_pass2 -1) + 1

                # Process and Draw Keypoints (same logic as before)
                if 'keypoints' in frame_data and frame_data['keypoints'] and len(frame_data['keypoints']) > 0:
                    kpts_list = frame_data['keypoints'][0]
                    if isinstance(kpts_list, list) and len(kpts_list) > 0:
                        keypoints_np = np.array(kpts_list).reshape(-1, 3)
                        valid_kpts = keypoints_np[keypoints_np[:, 2] > 0.2].copy()
                        if valid_kpts.size > 0:
                             # (Transformation logic - same as before)
                             min_x, max_x = np.min(valid_kpts[:, 0]), np.max(valid_kpts[:, 0])
                             min_y, max_y = np.min(valid_kpts[:, 1]), np.max(valid_kpts[:, 1])
                             current_width = max_x - min_x
                             current_height = max_y - min_y
                             scale_x = max_width / current_width if current_width > 0 else 1.0
                             scale_y = max_height / current_height if current_height > 0 else 1.0
                             valid_kpts[:, 0] = (valid_kpts[:, 0] - min_x) * scale_x + padding
                             valid_kpts[:, 1] = (valid_kpts[:, 1] - min_y) * scale_y + padding

                             # (Annotation logic - same as before)
                             if pose_model:
                                 try:
                                      annotator = Annotator(frame, line_width=2)
                                      annotator.kpts(valid_kpts, shape=frame_size, kpt_line=True)
                                      frame = annotator.result()
                                 except Exception as annotate_err:
                                      logging.warning(f"Annotation failed frame {frame_write_counter}: {annotate_err}. Drawing dots.")
                                      for x, y, conf in valid_kpts: cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)
                             else: # Fallback drawing
                                 for x, y, conf in valid_kpts: cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                # Add Text Overlay (same as before)
                text = f"Class: {action_class}, File: {json_file_counter}/{len(json_files)}, Frame: {local_frame_number}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

                # Write Frame
                out.write(frame)
                frame_write_counter += 1
                if frame_write_counter % 50 == 0: logging.info(f"Frames written: {frame_write_counter}...")

        except Exception as e:
            logging.warning(f"Video Gen Warning: Error during frame creation from {filepath}: {e}", exc_info=False)
            continue

    # --- Finalize Video ---
    if frame_write_counter == 0:
        logging.error(f"Video Gen Error: No frames written for {action_class}. Aborting.")
        out.release()
        try:
            if os.path.exists(output_video_path): os.remove(output_video_path)
        except OSError as e: logging.warning(f"Could not remove empty file {output_video_path}: {e}")
        return False

    logging.info(f"Releasing VideoWriter. Total frames written: {frame_write_counter}")
    out.release()

    # Final check
    if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
        logging.info(f"Video saved successfully: {output_video_path}, Size: {os.path.getsize(output_video_path)} bytes")
        return True
    else:
        logging.error(f"Video Gen Error: Output file missing/empty AFTER release: {output_video_path}")
        try:
             if os.path.exists(output_video_path): os.remove(output_video_path)
        except OSError as e: logging.warning(f"Could not remove broken file {output_video_path}: {e}")
        return False


# --- Main EDA Function ---

def perform_eda(dataset_id: str):
    """
    Performs EDA on the dataset specified by dataset_id, logging results to ClearML.
    """
    task = Task.current_task()
    logger = task.get_logger() if task else None
    if task: logging.info(f"Running EDA within ClearML Task: {task.id}")
    else: logging.warning("Not running within a ClearML Task context. Reporting disabled.")

    logging.info(f"Starting EDA process for dataset_id: {dataset_id}")

    # --- Get Dataset ---
    # (Remains the same)
    local_dataset_path = None
    try:
        dataset = Dataset.get(dataset_id=dataset_id)
        logging.info(f"Retrieved dataset object: {dataset.name} version {dataset.version}")
        local_dataset_path = dataset.get_local_copy()
        logging.info(f"Dataset obtained locally at: {local_dataset_path}")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to get/download dataset {dataset_id}: {e}", exc_info=True)
        raise

    # --- Run JSON Count Plot ---
    # (Remains the same)
    logging.info("--- Starting JSON Count ---")
    action_classes = ["Falling", "No Action", "Waving"]
    root_dirs_for_plot = [os.path.join(local_dataset_path, action) for action in action_classes]
    count_plot_fig = plot_json_counts(root_dirs_for_plot, action_classes)
    if count_plot_fig and logger:
        try:
             logger.report_matplotlib_figure(
                title="JSON File Counts per Class", series="EDA Plots", figure=count_plot_fig, report_image=True
            )
             logging.info("JSON count plot reported.")
        except Exception as report_err: logging.error(f"Failed to report JSON count plot: {report_err}", exc_info=True)
    elif not count_plot_fig: logging.warning("JSON count plot was not generated.")
    if count_plot_fig: plt.close(count_plot_fig)
    logging.info("--- JSON Count Finished ---")


    # --- Run Keypoint Video Generation ---
    logging.info("\n--- Starting Keypoint Video Generation ---")
    video_generation_failed = False
    for action_class in action_classes:
        class_folder_path = os.path.join(local_dataset_path, action_class)
        # *** CHANGE: Use .mp4 extension ***
        output_video_path = f"{action_class}_keypoints_visualization.mp4"
        logging.info(f"--- Generating video for class: {action_class} ---")

        if os.path.isdir(class_folder_path):
            # Call the video generation function (which now uses mp4v)
            success = keypoints_folder_to_video(
                folder_path=class_folder_path,
                output_video_path=output_video_path, # Pass the .mp4 path
                fps=10.0,
                yolo_model_path="yolo11n-pose.pt" # Keep model name for now
            )
            logging.info(f"Result returned by keypoints_folder_to_video for {action_class}: {success}")

            # Check file validity
            final_success = False
            if success:
                if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
                    logging.info(f"Video file confirmed valid: {output_video_path}, Size: {os.path.getsize(output_video_path)} bytes")
                    final_success = True
                else:
                    logging.error(f"Video function returned True, but file invalid/missing: {output_video_path}")

            # Report if valid and logger exists
            if final_success and logger:
                logging.info(f"Uploading video artifact: {output_video_path}")
                try:
                    logger.report_media(
                        title=f"{action_class} Keypoint Visualization",
                        series="EDA Videos",
                        local_path=output_video_path
                    )
                except Exception as report_err:
                    logging.error(f"Failed to report video {output_video_path}: {report_err}", exc_info=True)

            # Track overall failure
            if not final_success:
                 logging.error(f"Video generation process failed for class: {action_class}")
                 video_generation_failed = True
        else:
            logging.warning(f"Directory not found for class {action_class}. Skipping video generation.")

    logging.info("--- Keypoint Video Generation Finished ---")

    if video_generation_failed:
         logging.warning("One or more video generations failed during EDA.")
         # raise RuntimeError("EDA Step failed due to video generation errors.")

    logging.info(f"EDA process completed for dataset_id: {dataset_id}")


# --- Optional: Make script runnable standalone ---
# (if __name__ == '__main__': block remains the same)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
    parser = argparse.ArgumentParser(description="Run Dataset EDA (Counts and Videos).")
    parser.add_argument("--dataset-id", required=True, help="ClearML Dataset ID to process.")
    parser.add_argument("--action-class", default=None, help="Optional: Run video generation for only this action class.")
    args = parser.parse_args()
    logging.info(f"Running EDA standalone for Dataset ID: {args.dataset_id}")
    # (Standalone execution logic remains same)
    try:
        perform_eda(dataset_id=args.dataset_id)
        logging.info("Standalone EDA finished successfully.")
    except Exception as e:
        logging.error(f"Standalone EDA failed: {e}", exc_info=True)
        import sys
        sys.exit(1)
