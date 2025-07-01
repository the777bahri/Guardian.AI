import os
import re
import torch
# Imports for summary/plots
from torchinfo import summary as torchinfo_summary # Avoid name clash
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import logging # Keep logging import in case it's used elsewhere or for future use
from clearml import Task, Dataset, PipelineController, OutputModel  # Keep PipelineController for potential future use
from clearml.automation.controller import PipelineDecorator
from pose_dataset import PoseDataset
from model import ActionRecognitionBiLSTMWithAttention
import time # Keep time import for potential future use
import torch.optim as optim
import torch.nn as nn # Keep nn import for potential future use
from datetime import datetime # Keep datetime import for potential future use
import sys # Import sys for exiting on error

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_clearml_conf(conf_path):
    """
    Loads ClearML configuration from a file and sets environment variables.

    Args:
        conf_path (str): The path to the ClearML configuration file.

    Returns:
        bool: True if configuration was loaded successfully, False otherwise.
    """
    try:
        with open(conf_path, "r") as file:
            content = file.read()

        # Extract keys using regex with checks for matches
        access_key_match = re.search(r'access_key"\s*=\s*"([^"]+)"', content)
        secret_key_match = re.search(r'secret_key"\s*=\s*"([^"]+)"', content)
        api_host_match = re.search(r'api_server:\s*(https?://[^\s]+)', content)
        web_host_match = re.search(r'web_server:\s*(https?://[^\s]+)', content)
        files_host_match = re.search(r'files_server:\s*(https?://[^\s]+)', content)

        if not all([access_key_match, secret_key_match, api_host_match, web_host_match, files_host_match]):
            missing = []
            if not access_key_match: missing.append("access_key")
            if not secret_key_match: missing.append("secret_key")
            if not api_host_match: missing.append("api_server")
            if not web_host_match: missing.append("web_server")
            if not files_host_match: missing.append("files_server")
            logging.error(f"Could not find the following keys in {conf_path}: {', '.join(missing)}")
            return False

        access_key = access_key_match.group(1)
        secret_key = secret_key_match.group(1)
        api_host = api_host_match.group(1)
        web_host = web_host_match.group(1)
        files_host = files_host_match.group(1)

        os.environ["CLEARML_API_ACCESS_KEY"] = access_key
        os.environ["CLEARML_API_SECRET_KEY"] = secret_key
        os.environ["CLEARML_API_HOST"] = api_host
        os.environ["CLEARML_WEB_HOST"] = web_host
        os.environ["CLEARML_FILES_HOST"] = files_host
        logging.info("ClearML configuration loaded successfully.")
        return True

    except FileNotFoundError:
        logging.error(f"Configuration file not found: {conf_path}")
        return False
    except Exception as e:
        logging.error(f"An error occurred while loading ClearML config: {e}")
        return False











# --- ClearML Pipeline Component ---
# Note: Removed 'dataset_local_path' from return_values as it's not explicitly returned.
# ClearML might provide ways to access component artifacts/paths if needed later.
@PipelineDecorator.component(return_values=["dataset_id", "dataset_root_path"])
def upload_pose_dataset():
    """
    Versions a dataset in ClearML, uploading a new version only if changes are detected.
    If no changes are detected, it reuses the existing dataset version.
    This function creates a new dataset definition, checks for changes in the dataset files, 
    and uploads the dataset if necessary.
    It also handles the case where no previous dataset exists.
    If the dataset upload fails, it attempts to clean up the failed dataset object in ClearML.

    Returns:
        str: The ClearML Dataset ID (either existing or newly created). None if an error occurs.
    """
    dataset_name = "Guardian_Dataset"
    dataset_project = "Guardian_Training"
    dataset_tags = ["version1"] # Consider making tags more dynamic if needed

    dataset_root_path = r"data/" # This was unused, using dataset_root_path arg now

    # No need to get Task.current_task() if not used
    # task = Task.current_task()

    # Create dataset definition (metadata only at this point)
    try:
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=dataset_tags
        )
    except Exception as e:
        logging.error(f"Failed to initiate dataset creation in ClearML: {e}")
        return None # Return None to indicate failure

    logging.info("Checking for dataset changes...")

    prev_dataset = None
    try:
        # Get the latest completed version if it exists
        # Using allow_empty=True to avoid error if no dataset exists yet
        prev_dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=True,
            # allow_empty=True # Add this to handle the case where no dataset exists yet
        )
    except Exception as e:
        # Log the error but proceed, assuming no previous dataset exists
        logging.warning(f"Could not retrieve previous dataset (perhaps it's the first run?): {e}")


    dataset_id = None # Initialize dataset_id

    if prev_dataset:
        logging.info(f"Found previous dataset version: {prev_dataset.id}")

        # Get the list of files from the local path
        current_files = set()
        if not os.path.isdir(dataset_root_path):
             logging.error(f"Local dataset path does not exist or is not a directory: {dataset_root_path}")
             # Abort the dataset object locally before returning
             try:
                 dataset.delete(force=True)
                 logging.info("Aborted local dataset creation due to missing local data.")
             except Exception as delete_err:
                 logging.error(f"Error trying to abort local dataset object: {delete_err}")
             return None # Return None to indicate failure

        for root, _, files in os.walk(dataset_root_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Create relative path based on the provided dataset_root_path
                rel_path = os.path.relpath(file_path, dataset_root_path)
                # Use forward slashes for consistency, as ClearML often uses them internally
                current_files.add(rel_path.replace("\\", "/"))

        # Get the list of files from the previous ClearML dataset
        prev_files = set()
        try:
            # list_files() returns relative paths already
            prev_files = set(prev_dataset.list_files())
        except Exception as e:
            logging.error(f"Could not list files from previous dataset {prev_dataset.id}: {e}")
            # Decide if you want to proceed or return failure
            # Proceeding cautiously, assuming change check failed -> force upload
            prev_dataset = None # Force creation of new version if listing files fails

        # Check if files have changed
        if prev_dataset and current_files == prev_files:
            logging.info("No changes detected in dataset files. Reusing existing dataset version.")
            dataset_id = prev_dataset.id
            # We don't need the new 'dataset' object, maybe clean it up?
            # ClearML might handle orphaned dataset objects, but explicit is safer.
            try:
                dataset.delete(force=True)
                logging.info(f"Deleted unused new dataset definition, reusing {dataset_id}.")
            except Exception as delete_err:
                logging.error(f"Could not delete unused dataset definition: {delete_err}")
            # Return the existing ID
            return dataset_id
        else:
             if prev_dataset:
                 logging.info("Changes detected compared to previous dataset version.")
             else:
                 logging.info("No previous completed dataset found or error listing files, creating new version.")

    # --- This block runs if no previous dataset OR if changes were detected ---
    try:
        logging.info(f"Adding files from '{dataset_root_path}' to the new dataset version...")
        # Add files from the specified local path
        dataset.add_files(path=dataset_root_path)

        logging.info("Uploading dataset files to ClearML storage...")
        # Upload the added files (no arguments needed after add_files)
        # Use `output_url` if you need to specify a specific storage location
        dataset.upload() # Removed the incorrect dataset_path argument

        # Finalize the dataset (makes it immutable and usable)
        dataset.finalize()

        # Assign the ID *after* successful creation and upload
        dataset_id = dataset.id
        logging.info(f"Successfully created and uploaded new dataset version. Dataset ID: {dataset_id}")
        return dataset_id, dataset_root_path

    except Exception as e:
        logging.error(f"Failed to add, upload, or finalize the new dataset version: {e}")
        # Attempt to clean up the failed dataset object in ClearML
        try:
            dataset.delete(force=True)
            logging.info("Deleted failed dataset version from ClearML.")
        except Exception as delete_err:
            logging.error(f"Could not delete failed dataset object: {delete_err}")
        return None # Return None to indicate failure






@PipelineDecorator.component(name="Data EDA") # Name shown in UI
def run_data_eda_component(dataset_id: str):
    """
    Wrapper component that executes the external datasetEDA.py script.
    """
    # Make sure datasetEDA.py is accessible in the execution environment
    # ClearML usually handles this if it's in the same tracked Git repository.
    try:
        # Import the main function from your external script
        from datasetEDA import perform_eda
    except ImportError as e:
        logging.error(f"Failed to import perform_eda from datasetEDA.py. "
                      f"Ensure the file exists and is accessible. Error: {e}", exc_info=True)
        raise # Fail the component if the script can't be imported

    logging.info(f"Executing EDA function from datasetEDA.py for dataset_id: {dataset_id}")
    try:
        # Call the imported function
        perform_eda(dataset_id=dataset_id)
        logging.info("External EDA script execution completed.")
        # No explicit return needed unless perform_eda returns a value you need later
    except Exception as e:
        # Catch errors from the external script execution
        logging.error(f"An error occurred during the execution of perform_eda: {e}", exc_info=True)
        raise # Re-raise the exception to make the pipeline step fail


























@PipelineDecorator.component(
    name="Prepare DataLoaders with Logging", # Descriptive name
    return_values=["train_loader", "test_loader", "input_size", "num_classes", "action_classes"],
    packages=[ # Explicitly list dependencies
        'torch>=1.9',
        'scikit-learn',
        'clearml',
        'numpy' # Assuming PoseDataset might use numpy
        # Add 'pose_dataset' package if applicable
        ]
    # Note about returning DataLoaders still applies
)
def prepare_data(dataset_root_path: str = None): # Allow None for default
    """
    Loads data using PoseDataset from dataset_root_path, logs parameters & stats
    to ClearML, splits data, creates DataLoaders.

    Args:
        dataset_root_path (str, optional): The local filesystem path to the root of the dataset.
                                           Defaults to './data' if None.

    Returns:
        tuple: (train_loader, test_loader, input_size, num_classes)
    """
    # ===== Imports needed ONLY inside the component scope =====
    # (Moved necessary imports inside, standard practice for components)
    import os
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from clearml import Task
    import logging # Use logging inside too

    # --- Setup: Get Task, Logger, Handle Default Path ---
    task = Task.current_task()
    logger = task.get_logger() if task else None
    if task:
        logging.info(f"Running data preparation within ClearML Task: {task.id}")
    else:
        logging.warning("Could not get current ClearML Task. ClearML reporting will be limited.")

    # Handle default path
    original_path_arg = dataset_root_path # Store original argument for logging
    if dataset_root_path is None:
        dataset_root_path = "data/" # Default path if not provided
        logging.info(f"dataset_root_path not provided, using default: '{dataset_root_path}'")
    else:
        logging.info(f"Using provided dataset_root_path: '{dataset_root_path}'")

    # Connect input arguments to ClearML Task (even if default was used)
    if task:
        task.connect({"input_dataset_root_path": original_path_arg,
                      "effective_dataset_root_path": dataset_root_path},
                     name='Input Path Configuration')

    # --- Validate Path ---
    logging.info("Validating dataset path...")
    if not os.path.isdir(dataset_root_path):
         logging.error(f"CRITICAL: Effective dataset path is not a valid directory: '{dataset_root_path}'")
         raise NotADirectoryError(f"Effective dataset path not found or is not a directory: {dataset_root_path}")
    logging.info("Dataset path validated.")

    # --- Load Full Dataset ---
    action_classes = ["Falling", "No Action", "Waving"]
    logging.info(f"Defined action classes: {action_classes}")
    if task:
        task.connect({"action_classes": action_classes}, name='Class Configuration')

    logging.info(f"Loading full dataset using PoseDataset from path: '{dataset_root_path}'")
    try:
        # CORE LOGIC - KEPT INTACT
        dataset = PoseDataset(dataset_root_path, action_classes)
        # ---------

        num_total_samples = len(dataset)
        logging.info(f"Full dataset loaded. Total samples found: {num_total_samples}")
        if logger:
            logger.report_text(f"Loaded {num_total_samples} total samples.")

        if not hasattr(dataset, 'data') or not hasattr(dataset, 'labels'):
             logging.error("Loaded PoseDataset object missing '.data' or '.labels' attributes.")
             raise AttributeError("PoseDataset missing .data or .labels")
        data_to_split = dataset.data
        labels_to_split = dataset.labels
        logging.info(f"Data type for splitting: {type(data_to_split)}, "
                     f"Labels type: {type(labels_to_split)}")
        # Log shape only if it's meaningful (e.g., numpy array, tensor)
        if hasattr(data_to_split, 'shape'):
             logging.info(f"Data shape before splitting: {data_to_split.shape}")

    except Exception as e:
        logging.error(f"Failed to load data using PoseDataset from '{dataset_root_path}': {e}", exc_info=True)
        raise

    # --- Split Data ---
    split_params = {'test_size': 0.2, 'random_state': 42, 'stratify': True}
    logging.info(f"Splitting data with parameters: {split_params}")
    if task:
        task.connect(split_params, name='Data Split Config')

    try:
        # CORE LOGIC - KEPT INTACT
        stratify_param = labels_to_split if split_params['stratify'] else None
        train_data, test_data, train_labels, test_labels = train_test_split(
            data_to_split, labels_to_split,
            test_size=split_params['test_size'],
            random_state=split_params['random_state'],
            stratify=stratify_param
        )
        # ---------
        num_train_samples = len(train_labels)
        num_test_samples = len(test_labels)
        print(f"Train samples: {num_train_samples}, Test samples: {num_test_samples}")
        print("="*20 +"\n")
        logging.info(f"Data split complete. Train samples: {num_train_samples}, Test samples: {num_test_samples}")
        if logger:
            logger.report_text(f"Split results: Train={num_train_samples}, Test={num_test_samples}")
            # Log split sizes as scalars for potential plotting
            logger.report_scalar(title="Dataset Split Size", series="Train", value=num_train_samples, iteration=0)
            logger.report_scalar(title="Dataset Split Size", series="Test", value=num_test_samples, iteration=0)

    except Exception as e:
        logging.error(f"An unexpected error occurred during train_test_split: {e}", exc_info=True)
        raise

    # --- Create Train/Test PoseDataset Instances ---
    logging.info("Creating specific train/test PoseDataset instances...")
    try:
        # CORE LOGIC - KEPT INTACT
        train_dataset = PoseDataset(dataset_root_path, action_classes) # Pass path if constructor needs it
        train_dataset.data = train_data
        train_dataset.labels = train_labels

        test_dataset = PoseDataset(dataset_root_path, action_classes) # Pass path if constructor needs it
        test_dataset.data = test_data
        test_dataset.labels = test_labels
        # ---------
        logging.info(f"Train dataset instance created (length: {len(train_dataset)}).")
        logging.info(f"Test dataset instance created (length: {len(test_dataset)}).")
    except Exception as e:
         logging.error(f"Failed to create train/test PoseDataset instances: {e}", exc_info=True)
         raise


    # --- Create DataLoaders ---
    loader_params = {'batch_size': 32, 'num_workers': 0, 'shuffle_train': True} # Set parameters
    logging.info(f"Creating DataLoaders with parameters: {loader_params}")
    if task:
        task.connect(loader_params, name='DataLoader Config')

    try:
        # CORE LOGIC - KEPT INTACT
        train_loader = DataLoader(train_dataset, batch_size=loader_params['batch_size'], shuffle=loader_params['shuffle_train'], num_workers=loader_params['num_workers'])
        test_loader = DataLoader(test_dataset, batch_size=loader_params['batch_size'], shuffle=False, num_workers=loader_params['num_workers'])
        # ---------
        logging.info("Train and Test DataLoaders created successfully.")
    except Exception as e:
        logging.error(f"Failed to create DataLoaders: {e}", exc_info=True)
        raise

    # --- Determine Input Size and Number of Classes ---
    # CORE LOGIC - KEPT INTACT
    input_size = 34  # Example: 17 keypoints x 2 (x, y) - Confirm this is correct for your data
    num_classes = len(action_classes)
    # ---------
    output_dims = {'input_features': input_size, 'output_classes': num_classes}
    logging.info(f"Determined model input/output dimensions: {output_dims}")
    if task:
        task.connect(output_dims, name='Model Dimensions')


    logging.info("Data preparation component finished successfully.")

    # --- Return Results ---
    # CORE LOGIC - KEPT INTACT
    return train_loader, test_loader, input_size, num_classes, action_classes

















# --- Training Component Definition ---
# In your main pipeline script or datasetEDA.py...

@PipelineDecorator.component(
    name="Train Model and Log Scalars", # New name
    # Return path and history lists
    return_values=[
        "local_model_path",
        "train_loss_hist", "train_acc_hist",
        "test_loss_hist", "test_acc_hist"
        ],
    packages=[
        'torch>=1.9', 'clearml', 'scikit-learn', 'numpy',
        'ultralytics', 'opencv-python'
    ],
    task_type='training',
    cache=False
)
def train_bilstm(
    train_loader, test_loader, # Include test_loader for getting history
    input_size: int, num_classes: int,
    base_lr: float = 0.001, epochs: int = 50,
    hidden_size: int = 256, num_layers: int = 4
    ):
    """
    Trains model, logs SCALARS (loss/acc) per epoch to ClearML, saves model
    LOCALLY, and returns the local path and metric history lists.
    """
    # ===== Imports needed inside component =====
    import torch.nn as nn
    import torch.optim as optim
    from clearml import Task
    import logging
    import time
    # --- ClearML Setup (for SCALAR logging ONLY) ---
    task = Task.current_task()
    logger = task.get_logger() if task else None
    task_id = task.id if task else "local_run"
    if task:
        logging.info(f"Running training within ClearML Task: {task_id}.")
        # Connect hyperparameters
        hyperparams = {
            'base_lr': base_lr, 'epochs': epochs, 'hidden_size': hidden_size,
            'num_layers': num_layers, 'input_size': input_size, 'num_classes': num_classes,
            'batch_size': train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'N/A'
        }
        task.connect(hyperparams, name='Hyperparameters')
        logging.info(f"Connected hyperparameters: {hyperparams}")
    else:
        logging.warning("ClearML task context not found. Scalar logging disabled.")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if logger: logger.report_text(f"Using device: {device}")

    # --- Model, Optimizer, Loss ---
    logging.info("Initializing model, optimizer, and loss function...")
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size, hidden_size=hidden_size,
        num_layers=num_layers, num_classes=num_classes
    ).to(device) # Ensure signature matches model.py
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    criterion = nn.CrossEntropyLoss()
    logging.info("Model, optimizer, and loss function initialized.")

    if task: # Log only if running in ClearML context
         try:
             # Calculate input size based on expected tensor shape (Seq_Len, Batch?, Features)
             # Assuming batch_size doesn't affect layer definition, use 1
             # Assuming max_frames was used in PoseDataset, use that. Need max_frames here!
             # Let's assume max_frames=40 as per PoseDataset default
             # If PoseDataset max_frames changes, this needs updating or passing max_frames
             example_input_shape = (1, 40, input_size) # (Batch, Seq, Features)
             model_summary = torchinfo_summary(model, input_size=example_input_shape, verbose=0) # verbose=0 prevents printing to console
             summary_str = str(model_summary)
             logging.info("\nModel Summary:\n" + summary_str)
             logger.report_text(summary_str, name="Model Summary", print_console=False)
         except Exception as summary_err:
             logging.warning(f"Could not generate torchinfo summary: {summary_err}")

         try:
            arch_str = str(model)
            logging.info("\nModel Architecture:\n" + arch_str)
            logger.report_text(arch_str, name="Model Architecture", print_console=False)
         except Exception as arch_err:
            logging.warning(f"Could not get model string representation: {arch_err}")

    # --- Pre-Loop Checks ---
    logging.info("Checking data loaders...")
    if not train_loader or not hasattr(train_loader, 'dataset') or len(train_loader.dataset) == 0: return None, [], [], [], [] # Return empty lists on failure
    if not test_loader or not hasattr(test_loader, 'dataset') or len(test_loader.dataset) == 0: return None, [], [], [], [] # Return empty lists on failure
    logging.info(f"Train loader: {len(train_loader)} batches, {len(train_loader.dataset)} samples.")
    logging.info(f"Test loader: {len(test_loader)} batches, {len(test_loader.dataset)} samples.")

    # --- Lists to store history for plotting later ---
    train_loss_hist, train_acc_hist = [], []
    test_loss_hist, test_acc_hist = [], []
    # Store last epoch preds/labels for CM plots
    last_epoch_train_preds, last_epoch_train_labels = [], []
    last_epoch_test_preds, last_epoch_test_labels = [], []
    local_model_path = None # Initialize

    # --- Training & Evaluation Loop ---
    if epochs <= 0:
        logging.warning(f"Epoch count ({epochs}) not positive. Skipping training.")
    else:
        logging.info(f"Starting training & evaluation loop for {epochs} epochs...")
        for epoch in range(epochs):
            epoch_start_time = time.time()
            is_last_epoch = (epoch == epochs - 1) # Check if this is the last epoch
            logging.info(f"--- Starting Epoch {epoch+1}/{epochs} ---")

            # Train Phase
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0
            if is_last_epoch: last_epoch_train_preds.clear(); last_epoch_train_labels.clear()
            # ... (Inner training loop - same as before) ...
            try:
                 for keypoints, labels in train_loader:
                     keypoints, labels = keypoints.to(device).float(), labels.to(device)
                     outputs, _ = model(keypoints); loss = criterion(outputs, labels)
                     optimizer.zero_grad(); loss.backward(); optimizer.step()
                     train_loss += loss.item() * keypoints.size(0)
                     # ... (accumulate train metrics) ...
                     _, predicted = torch.max(outputs.data, 1)
                     if is_last_epoch: # Collect for CM plot on last epoch
                         last_epoch_train_preds.extend(predicted.cpu().numpy())
                         last_epoch_train_labels.extend(labels.cpu().numpy())
                     train_total += labels.size(0)
                     train_correct += (predicted == labels).sum().item()
            except Exception as loop_err: raise loop_err

            avg_train_loss = train_loss / train_total if train_total > 0 else 0
            train_accuracy = 100 * train_correct / train_total if train_total > 0 else 0
            train_loss_hist.append(avg_train_loss)
            train_acc_hist.append(train_accuracy)

            # Eval Phase
            model.eval()
            test_loss, test_correct, test_total = 0.0, 0, 0
            if is_last_epoch: last_epoch_test_preds.clear(); last_epoch_test_labels.clear()
            with torch.no_grad():
                 try:
                     for keypoints, labels in test_loader:
                         keypoints, labels = keypoints.to(device).float(), labels.to(device)
                         outputs, _ = model(keypoints); loss = criterion(outputs, labels)
                         test_loss += loss.item() * keypoints.size(0)
                         _, predicted = torch.max(outputs.data, 1)
                         if is_last_epoch: # Collect for CM plot on last epoch
                             last_epoch_test_preds.extend(predicted.cpu().numpy())
                             last_epoch_test_labels.extend(labels.cpu().numpy())
                         test_total += labels.size(0)
                         test_correct += (predicted == labels).sum().item()
                 except Exception as eval_err:
                     logging.error(f"Error during eval loop epoch {epoch+1}: {eval_err}", exc_info=True)
                     test_total = 0 # Invalidate test metrics

            avg_test_loss = test_loss / test_total if test_total > 0 else 0
            test_accuracy = 100 * test_correct / test_total if test_total > 0 else 0
            test_loss_hist.append(avg_test_loss)
            test_acc_hist.append(test_accuracy)

            epoch_duration = time.time() - epoch_start_time

            # Log SCALARS to ClearML
            if logger:
                logger.report_scalar("Loss", "Train", avg_train_loss, epoch)
                logger.report_scalar("Accuracy", "Train", train_accuracy, epoch)
                if test_total > 0:
                    logger.report_scalar("Loss", "Test", avg_test_loss, epoch)
                    logger.report_scalar("Accuracy", "Test", test_accuracy, epoch)
                logger.report_scalar("Epoch Duration", "Seconds", epoch_duration, epoch)

            # Log epoch summary to console
            logging.info(f"[Epoch {epoch+1}/{epochs}] Duration: {epoch_duration:.2f}s")
            logging.info(f"  Train -> Loss: {avg_train_loss:.4f}, Acc: {train_accuracy:.2f}%")
            if test_total > 0: logging.info(f"  Test  -> Loss: {avg_test_loss:.4f}, Acc: {test_accuracy:.2f}%")
            else: logging.info("  Test  -> Skipped or failed.")

        logging.info("Training loop finished.")

        # --- Generate and Log Confusion Matrices (After Loop) ---
        if logger and epochs > 0: # Check if training actually ran
            # NOTE: Needs action_classes, let's hardcode or pass it? Hardcoding for now.
            action_classes_names = ["Falling", "No Action", "Waving"] # Ideally pass this
            logging.info("Generating confusion matrices for last epoch...")

            # Training CM
            if last_epoch_train_labels and last_epoch_train_preds:
                try:
                    cm_train = confusion_matrix(last_epoch_train_labels, last_epoch_train_preds)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Greens', # Different color
                                xticklabels=action_classes_names, yticklabels=action_classes_names)
                    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Training Confusion Matrix (Last Epoch)")
                    plt.tight_layout()
                    logger.report_matplotlib_figure(title="Confusion Matrix", series="Train (Last Epoch)", figure=plt, report_image=True, iteration=epochs-1)
                    plt.close()
                    logging.info("Training confusion matrix reported.")
                except Exception as cm_err: logging.error(f"Failed train CM: {cm_err}", exc_info=True)
            else: logging.warning("No train preds/labels for CM.")

            # Test CM
            if last_epoch_test_labels and last_epoch_test_preds:
                 try:
                    cm_test = confusion_matrix(last_epoch_test_labels, last_epoch_test_preds)
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', # Keep blue for test
                                xticklabels=action_classes_names, yticklabels=action_classes_names)
                    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Test Confusion Matrix (Last Epoch)")
                    plt.tight_layout()
                    logger.report_matplotlib_figure(title="Confusion Matrix", series="Test (Last Epoch)", figure=plt, report_image=True, iteration=epochs-1)
                    plt.close()
                    logging.info("Test confusion matrix reported.")
                 except Exception as cm_err: logging.error(f"Failed test CM: {cm_err}", exc_info=True)
            else: logging.warning("No test preds/labels for CM.")


        # --- Save Model Locally ONLY ---
        model_filename = f"trained_model_{task_id}.pt" # Unique-ish local name
        local_model_path = os.path.abspath(model_filename) # Get full path
        logging.info(f"Saving final model state_dict locally to: {local_model_path}")
        try:
            torch.save(model.state_dict(), local_model_path)
            logging.info("Model saved successfully locally.")
        except Exception as save_err:
            logging.error(f"Failed to save model locally to {local_model_path}: {save_err}", exc_info=True)
            local_model_path = None # Indicate failure

        # --- NO OutputModel or artifact upload here ---

    logging.info(f"Training component finished. Returning Local Path: {local_model_path}")
    # Return path and history lists
    return local_model_path, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist








































# --- Evaluation Component Definition ---
# In your main pipeline script or a separate evaluation.py...

@PipelineDecorator.component(
    name="Evaluate Model and Plot Curves", # New name
    packages=[
        'torch>=1.9', 'clearml', 'scikit-learn', 'seaborn',
        'matplotlib', 'numpy', 'opencv-python'
    ],
    task_type='testing'
    # No return value needed
)
def evaluate_model(
    local_model_path: str,  # Received from train_bilstm
    test_loader,            # Received from prepare_data
    input_size: int,        # Needed to reconstruct model
    num_classes: int,       # Needed to reconstruct model
    hidden_size: int,       # Needed to reconstruct model
    num_layers: int,        # Needed to reconstruct model
    # History lists received from train_bilstm
    train_loss_hist: list,
    train_acc_hist: list,
    test_loss_hist: list,
    test_acc_hist: list,
    action_classes: list # Received from prepare_data
    # Note: action_classes is used for confusion matrix labels
    ):
    """
    Loads model from local path, generates learning curve plots from history,
    evaluates on test set, logs classification report and confusion matrix,
    and uploads plots/reports as ClearML artifacts.
    """
    # ===== Imports needed inside component =====
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg') # Use Agg backend
    import matplotlib.pyplot as plt
    import numpy as np
    from clearml import Task # Need Task for logging/artifacts
    import logging
    import os

    # --- ClearML Setup ---
    task = Task.current_task()
    logger = task.get_logger() if task else None
    if task:
        logging.info(f"Running evaluation within ClearML Task: {task.id}")
        task.connect({ # Connect config
            "input_local_model_path": local_model_path, "input_size": input_size,
            "num_classes": num_classes, "hidden_size": hidden_size, "num_layers": num_layers,
            "batch_size": test_loader.batch_size if hasattr(test_loader, 'batch_size') else 'N/A'
            }, name='Evaluation Config')
    else:
        logging.warning("ClearML task context not found.")

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    if logger: logger.report_text(f"Evaluation using device: {device}")

    # --- Load Model from Local Path ---
    logging.info(f"Attempting to load model weights from local path: {local_model_path}")
    try:
        if not local_model_path or not os.path.exists(local_model_path):
            logging.error(f"Provided local_model_path is invalid or file does not exist: '{local_model_path}'")
            raise FileNotFoundError(f"Weights file not found at: {local_model_path}")

        model = ActionRecognitionBiLSTMWithAttention(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, num_classes=num_classes
        )
        model.load_state_dict(torch.load(local_model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded locally and set to evaluation mode successfully.")

    except Exception as e:
        logging.error(f"Failed to load model from local path {local_model_path}: {e}", exc_info=True)
        raise

    # --- Generate and Log Learning Curve Plots ---
    if logger and train_loss_hist and test_loss_hist: # Check if history exists
        try:
            epochs_list = list(range(len(train_loss_hist)))
            plt.figure(figsize=(10, 5))
            plt.plot(epochs_list, train_loss_hist, label='Train Loss')
            plt.plot(epochs_list, test_loss_hist, label='Test Loss')
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Loss Curve")
            plt.legend()
            plt.grid(True)
            # Report plot to ClearML Debug Samples / Plots
            logger.report_matplotlib_figure(title="Loss Curve", series="Train/Test Loss", figure=plt, report_image=True, iteration=0)
            plt.close()
            logging.info("Loss curve plot reported.")
        except Exception as plot_err:
            logging.error(f"Failed to generate/report loss curve plot: {plot_err}", exc_info=True)

    if logger and train_acc_hist and test_acc_hist:
        try:
            epochs_list = list(range(len(train_acc_hist)))
            plt.figure(figsize=(10, 5))
            plt.plot(epochs_list, train_acc_hist, label='Train Accuracy')
            plt.plot(epochs_list, test_acc_hist, label='Test Accuracy')
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy (%)")
            plt.title("Accuracy Curve")
            plt.legend()
            plt.grid(True)
            # Report plot to ClearML Debug Samples / Plots
            logger.report_matplotlib_figure(title="Accuracy Curve", series="Train/Test Accuracy", figure=plt, report_image=True, iteration=0)
            plt.close()
            logging.info("Accuracy curve plot reported.")
        except Exception as plot_err:
            logging.error(f"Failed to generate/report accuracy curve plot: {plot_err}", exc_info=True)


    # --- Run Evaluation on Test Loader (for final metrics/CM) ---
    logging.info("Running final inference on the test set for metrics...")
    all_preds = []
    all_labels = []
    # (Evaluation loop - same as before: iterate test_loader, predict, collect)
    with torch.no_grad():
        for keypoints, labels in test_loader:
             keypoints, labels = keypoints.to(device).float(), labels.to(device)
             outputs, _ = model(keypoints)
             _, preds = torch.max(outputs, 1)
             all_preds.extend(preds.cpu().numpy())
             all_labels.extend(labels.cpu().numpy())

    logging.info("Inference complete. Calculating final metrics...")

    if not all_labels or not all_preds:
        logging.error("No labels/predictions collected. Cannot generate report/CM.")
        return

    # --- Calculate and Log Final Metrics / Confusion Matrix ---
    # (Metrics calculation and logging: report_text, upload_artifact, report_matplotlib_figure for CM - same as before)
    try:
        target_names = action_classes if action_classes and len(action_classes) == num_classes else [f"Class_{i}" for i in range(num_classes)]
        logging.info(f"Using target names for report/CM: {target_names}")

        # classification report
        report = classification_report(all_labels, all_preds, target_names=target_names, digits=3)
        logging.info("\nClassification Report (Test Set):\n" + report)
        if logger:
            logger.report_text(report, level=logging.INFO, print_console=False)
            #  Log artifact upload attempt
            logging.info("Attempting to upload classification report as artifact...")
            try:
                task.upload_artifact(name="test_classification_report", artifact_object=report, extension_name=".txt")
                logging.info("Classification report uploaded successfully as artifact.")
            except Exception as art_err:
                logging.error(f"Failed to upload classification report artifact: {art_err}", exc_info=True)


        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel("Predicted Label"); plt.ylabel("True Label"); plt.title("Confusion Matrix (Test Set)")
        plt.tight_layout()
        if logger:
            logger.report_matplotlib_figure(title="Confusion Matrix", series="Test Set Evaluation", figure=plt, report_image=True)
        logging.info("Confusion matrix generated and reported.")
        plt.close()

        # Log overall metrics as scalars
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, fscore, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        logging.info(f"Overall Test Accuracy: {accuracy:.4f}")
        logging.info(f"Weighted Avg -> Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {fscore:.4f}")
        if logger:
             logger.report_scalar("Overall Test Metrics", "Accuracy", accuracy, 0)
             logger.report_scalar("Overall Test Metrics", "Precision (Weighted)", precision, 0)
             logger.report_scalar("Overall Test Metrics", "Recall (Weighted)", recall, 0)
             logger.report_scalar("Overall Test Metrics", "F1-Score (Weighted)", fscore, 0)

    except Exception as metric_err:
        logging.error(f"Failed during final metric calculation or reporting: {metric_err}", exc_info=True)

    logging.info("Evaluation component finished.")
























# ------------------------
# Pipeline flow function
# ------------------------
@PipelineDecorator.pipeline(
    name="Guardian_Pipeline",
    project="Guardian_Training",
    # version="1.0.0" # Optional: Version your pipeline definition
)
def run_pipeline():
    logging.basicConfig(level=logging.INFO)
    logging.info("Pipeline started...")

    # --- Config & Hyperparameters ---
    local_data_folder = "./data"
    training_epochs = 50
    learning_rate = 0.001
    lstm_hidden_size = 256
    lstm_num_layers = 4

    # --- Step 1: Dataset ---
    dataset_id, dataset_path_from_upload = upload_pose_dataset(dataset_root_path=local_data_folder)
    if not dataset_id: raise ValueError("Dataset step failed.")
    dataset_path_to_use = dataset_path_from_upload
    logging.info(f"Dataset step completed. Using Dataset ID: {dataset_id}, Path: {dataset_path_to_use}")

    # --- Step 2: EDA (Optional) ---
    run_data_eda_component(dataset_id=dataset_id)

    # --- Step 3: Prepare Data ---
    logging.info("Starting data preparation step...")
    train_loader, test_loader, input_size, num_classes, action_classes = prepare_data(
        dataset_root_path=dataset_path_to_use
    )
    logging.info("Data preparation step completed.")

    # --- Step 4: Train Model (Log Scalars, Save Locally) ---
    logging.info("Starting model training step (Log scalars, save local)...")
    # Capture local path and history lists
    local_model_path, train_loss_hist, train_acc_hist, test_loss_hist, test_acc_hist = train_bilstm(
        train_loader=train_loader,
        test_loader=test_loader, # Pass test_loader to get history
        input_size=input_size,
        num_classes=num_classes,
        base_lr=learning_rate,
        epochs=training_epochs,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers
    )
    if not local_model_path: # Check if training/saving failed
        raise RuntimeError("Training step failed to save model.")
    logging.info(f"Training step completed. Model saved locally at: {local_model_path}")

    # --- Step 5: Evaluate Model (Load Local, Plot Curves/Metrics) ---
    logging.info("Starting model evaluation step...")
    # Pass local path and history lists
    evaluate_model(
        local_model_path=local_model_path, # Pass the path
        test_loader=test_loader,
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=lstm_hidden_size,
        num_layers=lstm_num_layers,
        # Pass history lists
        train_loss_hist=train_loss_hist,
        train_acc_hist=train_acc_hist,
        test_loss_hist=test_loss_hist,
        test_acc_hist=test_acc_hist,
        action_classes=action_classes # Pass action classes for CM labels
    )
    logging.info("Evaluation step completed.")

    logging.info("Pipeline finished successfully.")

    

# ------------------------
# Main execution block
# ------------------------
if __name__ == "__main__":
    # Load configuration first
    if not load_clearml_conf(r"clearML/KongML.txt"):
    # if not load_clearml_conf(r"clearML/FVLEGION.txt"):
        logging.error("Exiting script due to ClearML configuration loading failure.")
        sys.exit(1) # Exit with a non-zero code to indicate error

    logging.info("Running Guardian 🦾 pipeline locally...")
    # run_locally() executes the pipeline defined by decorators in the current process
    PipelineDecorator.run_locally()

    # Start the pipeline execution
    run_pipeline()

    logging.info("Local pipeline execution complete.")