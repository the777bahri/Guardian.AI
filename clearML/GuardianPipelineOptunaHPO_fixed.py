import os
import sys
import shutil
import tempfile

def load_clearml_conf(conf_path: str) -> bool:
    """
    Set the CLEARML_CONFIG_FILE environment variable.
    Returns True if the config file exists.
    """
    if not os.path.exists(conf_path):
        print(f"DEBUG: Config file {conf_path} not found.")
        return False
    abs_conf_path = os.path.abspath(conf_path)
    print(f"DEBUG: Setting CLEARML_CONFIG_FILE to: {abs_conf_path}")
    os.environ['CLEARML_CONFIG_FILE'] = abs_conf_path
    # We expect ClearML components to pick this up when they initialize.
    return True

# Ensure current working directory is discoverable by components
# This needs to be early, but after load_clearml_conf sets the env var.
sys.path.append(os.getcwd())

# Copy the required module files to a temp directory that will be accessible to all tasks
temp_dir = os.path.join(tempfile.gettempdir(), "guardian_modules")
os.makedirs(temp_dir, exist_ok=True)

# Copy the necessary files
current_dir = os.getcwd()
clearml_dir = os.path.join(current_dir, "clearML")

# Copy model.py and pose_dataset.py
for module_file in ["model.py", "pose_dataset.py"]:
    src_file = os.path.join(clearml_dir, module_file)
    dst_file = os.path.join(temp_dir, module_file)
    try:
        shutil.copy2(src_file, dst_file)
        print(f"Copied {src_file} to {dst_file}")
    except Exception as e:
        print(f"Failed to copy {src_file}: {str(e)}")

# Add the temp directory to Python path
sys.path.append(temp_dir)
print(f"Added {temp_dir} to sys.path")

# It is important that load_clearml_conf is called before ClearML modules
# that might read the configuration are imported or used extensively.
# We call it first in __main__.

from clearml.automation import PipelineDecorator
from clearml import Task # Keep this import here if components use it directly

@PipelineDecorator.component(
    name="Upload_Pose_Dataset",
    return_values=["dataset_id", "dataset_root_path"],
    cache=True,
    packages=["clearml"]
)
def upload_pose_dataset(data_dir: str = "./data"):
    """
    Version or reuse the local pose data directory in ClearML.
    Returns the dataset ID and the local path for subsequent steps.
    """
    # Self-contained imports
    import os
    from clearml import Dataset
    # Try to reuse existing dataset
    try:
        ds = Dataset.get(
            dataset_name="PoseDataset", 
            dataset_project="Guardian_Training", 
            only_completed=True
        )
    except Exception:
        ds = Dataset.create(
            dataset_name="PoseDataset", 
            dataset_project="Guardian_Training"
        )
        ds.add_files(data_dir)
        ds.upload()
        ds.finalize()
    # Obtain local copy
    local_copy = ds.get_local_copy()
    if not local_copy or not os.path.isdir(local_copy):
        local_copy = ds.get_mutable_local_copy(target_folder=data_dir)
    return ds.id, local_copy


@PipelineDecorator.component(
    name="Train_BiLSTM",
    return_values=["task_id", "local_model_path"],
    packages=["torch>=1.9", "clearml", "scikit-learn", "numpy"],
    task_type=Task.TaskTypes.training,
    cache=False
)
def train_bilstm(
    dataset_root_path: str,
    input_size: int = 34,
    base_lr: float = 0.001,
    epochs: int = 50,
    hidden_size: int = 256,
    num_layers: int = 4,
    batch_size: int = 32,
    dropout_rate: float = 0.1
):
    """
    Train a BiLSTM model on pose data, log validation accuracy for HPO,
    and publish the best checkpoint.
    Returns ClearML task ID and local model path.
    """
    # Self-contained imports
    import sys, os
    import tempfile
    
    # Get the temp directory with our modules
    temp_dir = os.path.join(tempfile.gettempdir(), "guardian_modules")
    if os.path.exists(temp_dir) and temp_dir not in sys.path:
        sys.path.append(temp_dir)
        print(f"Added {temp_dir} to sys.path")
    
    print("DEBUG: sys.path =", sys.path)
    print("DEBUG: cwd =", os.getcwd())
    
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from clearml import Task, OutputModel, Model
    
    # Import modules from the temp directory
    sys.path.insert(0, temp_dir)
    try:
        from pose_dataset import PoseDataset
        from model import ActionRecognitionBiLSTMWithAttention
        print("Successfully imported modules from temp directory")
    except ImportError as e:
        print(f"Error importing modules from {temp_dir}: {e}")
        # List files in the temp directory
        print(f"Files in {temp_dir}:", os.listdir(temp_dir) if os.path.exists(temp_dir) else "Directory not found")
        raise
    
    # Hardcode the action classes to avoid serialization issues
    action_classes = ["Falling", "No Action", "Waving"]
    num_classes = len(action_classes)
    
    # Create the output directory in temp folder if it doesn't exist
    output_dir = os.path.join(tempfile.gettempdir(), "clearml_guardian_output")
    os.makedirs(output_dir, exist_ok=True)
    output_uri = "file:///" + output_dir.replace("\\", "/")
    print(f"DEBUG: Using output_uri: {output_uri}")

    # Parameter casting
    base_lr = float(base_lr)
    epochs = int(epochs)
    hidden_size = int(hidden_size)
    num_layers = int(num_layers)
    batch_size = int(batch_size)
    dropout_rate = float(dropout_rate)

    # Initialize the task with explicit output_uri
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="Guardian_Training",
            task_name="Train_BiLSTM",
            output_uri=output_uri  # Explicitly set the output URI
        )
        print(f"DEBUG: Created new task with ID {task.id} and output_uri={output_uri}")
    else:
        print(f"DEBUG: Using existing task with ID {task.id}")
        
    logger = task.get_logger()
    task_id = task.id

    # Register hyperparameters for HPO
    if task:
        task.connect({
            'General/base_lr': base_lr,
            'General/epochs': epochs,
            'General/hidden_size': hidden_size,
            'General/num_layers': num_layers,
            'General/batch_size': batch_size,
            'General/dropout_rate': dropout_rate
        })

    # Load data
    dataset = PoseDataset(dataset_root_path, action_classes)
    data, labels = dataset.data, dataset.labels
    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_ds = PoseDataset(dataset_root_path, action_classes)
    train_ds.data, train_ds.labels = train_data, train_labels
    test_ds = PoseDataset(dataset_root_path, action_classes)
    test_ds.data, test_ds.labels = test_data, test_labels

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_model_path = None

    # Train & validate
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, _ = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs, _ = model(x)
                preds = outputs.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = 100 * correct / max(total, 1)
        # Report for HPO
        if logger:
            logger.report_scalar("Accuracy", "Validation", val_acc, epoch)
        # Save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_path = f"best_model_{task_id}.pt"
            torch.save(model.state_dict(), best_model_path)    # Publish best model
    if best_model_path and task:
        print("Publishing the best model to ClearML...")
        try:
            out = OutputModel(task=task, name="BiLSTM_ActionRecognition", framework="PyTorch")
            out.update_weights(weights_filename=best_model_path)
            out.publish()
            print(f"Successfully published model with ID: {out.id}")
            
            # Wait to ensure model is fully registered in ClearML
            import time
            print("Waiting for model registration to complete...")
            time.sleep(5)
            
            # Verify model is retrievable
            test_model = Model.query_models(
                project_name=task.get_project_name(),
                model_name="BiLSTM_ActionRecognition",
                only_published=True,
                max_results=1
            )
            if test_model:
                print(f"Model verification successful. Found model ID: {test_model[0].id}")
            else:
                print("WARNING: Model verification failed - could not retrieve the published model via query")
                
        except Exception as e:
            print(f"ERROR publishing model: {e}")
            # Even if publishing fails, we still want to return the local path

    final_model_path = os.path.abspath(best_model_path) if best_model_path else None
    print(f"********************************** \n Final model path: {final_model_path}")
    
    if task:
        task.flush() # Ensure all data is sent to the server before task finishes
        print("Task data flushed to server")

    return task_id, final_model_path


@PipelineDecorator.component(
    name="BiLSTM_Optuna_HPO",
    return_values=["best_task_id", "best_model_id", "best_model_path"],
    cache=False,
    packages=["clearml", "optuna"]
)
def bilstm_hyperparam_optimizer(
    base_task_template_id: str,
    dataset_root_path: str,
    input_size: int,
    total_max_trials: int = 5,
    save_top_k: int = 3
):
    """
    Runs sequential Optuna HPO on the Train_BiLSTM component.
    Returns best task ID, model ID, and local model path.
    """
    import os
    import sys
    import tempfile
    from clearml import Task
    from clearml.automation import HyperParameterOptimizer, UniformParameterRange, UniformIntegerParameterRange, DiscreteParameterRange
    from clearml.automation.optuna import OptimizerOptuna
    from clearml import Model
    
    # Create the output directory in temp folder if it doesn't exist
    output_dir = os.path.join(tempfile.gettempdir(), "clearml_guardian_output")
    os.makedirs(output_dir, exist_ok=True)
    output_uri = "file:///" + output_dir.replace("\\", "/")
    print(f"DEBUG: HPO Using output_uri: {output_uri}")

    hpo_task = Task.init(
        project_name="Guardian_Training/HPO",
        task_name="BiLSTM_Optuna_HPO_Controller",
        task_type=Task.TaskTypes.optimizer,
        output_uri=output_uri,  # Explicitly set the output URI
        reuse_last_task_id=False
    )

    # Define search space
    space = [
        UniformParameterRange('General/base_lr', 1e-4, 1e-2),
        UniformIntegerParameterRange('General/hidden_size', 64, 256, step_size=64),
        UniformIntegerParameterRange('General/num_layers', 1, 3),
        UniformParameterRange('General/dropout_rate', 0.1, 0.5),
        DiscreteParameterRange('General/batch_size', values=[16,32,64]),
        DiscreteParameterRange('General/epochs', values=[10,20])
    ]
      # Define fixed arguments for child tasks
    fixed_args = {
        'dataset_root_path': dataset_root_path,
        'input_size': input_size
    }
    
    optimizer = HyperParameterOptimizer(
        min_iteration_per_job=1,  # Ensure at least one iteration completes
        max_iteration_per_job=None,
        base_task_id=base_task_template_id,
        hyper_parameters=space,
        objective_metric_title="Accuracy",
        objective_metric_series="Validation",
        objective_metric_sign="max",
        optimizer_class=OptimizerOptuna,
        total_max_jobs=total_max_trials,
        save_top_k_tasks_only=save_top_k,
        base_task_kwargs=fixed_args,
        time_limit_per_job=600,  # 10 minutes per job
        pool_period_min=0.1,  # Check status frequently
        max_failures_per_job=3  # Allow retries for failed jobs
    )
    
    print("Starting optimization process...")
    optimizer.start_locally(job_complete_callback=lambda x: print(f"Job completed: {x}"))
    print("Optimization process finished, stopping the optimizer...")
    optimizer.stop() # Stop the HPO controller task. Individual trials should have completed
    top_tasks = optimizer.get_top_experiments(top_k=1)
    if not top_tasks:
        raise RuntimeError("No HPO experiments returned by optimizer.")
    
    best_hpo_trial_task = top_tasks[0] # This is a clearml.Task object
    print(f"Best HPO trial task ID: {best_hpo_trial_task.id}")
    print(f"Best HPO trial task status: {best_hpo_trial_task.status}")    # Wait for the task to complete if it's still running
    if best_hpo_trial_task.status not in ['completed', 'stopped']:
        print(f"Waiting for best HPO trial task {best_hpo_trial_task.id} to complete...")
        best_hpo_trial_task.wait_for_status(status=['completed', 'stopped'], timeout=300)
        print(f"Best HPO trial task {best_hpo_trial_task.id} status is now: {best_hpo_trial_task.status}")
        
        # Reload the task data to get updated information
        best_hpo_trial_task = Task.get_task(task_id=best_hpo_trial_task.id)
    
    print("Beginning comprehensive model search process...")
    found_model = None
    expected_model_name = "BiLSTM_ActionRecognition"
    
    # Strategy 1: Try to get models directly from the best HPO trial task
    print("Strategy 1: Looking for models in the best HPO trial task...")
    output_models_from_task = best_hpo_trial_task.models.get('output', [])
    print(f"Found {len(output_models_from_task)} models in task output")
    
    for model_candidate in output_models_from_task:
        if model_candidate.name == expected_model_name:
            found_model = model_candidate
            print(f"Found matching model {found_model.id} directly in task output")
            break
    
    # Strategy 2: Check models linked to the task
    if not found_model:
        print("Strategy 2: Looking for models linked to the best HPO trial task...")
        # Reload the task to ensure we have the latest data
        refreshed_task = Task.get_task(task_id=best_hpo_trial_task.id)
        task_models = refreshed_task.get_models()
        print(f"Found {len(task_models)} models linked to the task")
        
        for model in task_models:
            if model.name == expected_model_name:
                found_model = model
                print(f"Found matching model {found_model.id} linked to the task")
                break
    
    # Strategy 3: Check task artifacts for model files
    if not found_model:
        print("Strategy 3: Checking task artifacts for model files...")
        artifacts = best_hpo_trial_task.artifacts
        print(f"Task artifacts: {list(artifacts.keys()) if artifacts else 'None'}")
        
        # Look for model file artifacts
        model_artifacts = [k for k in artifacts.keys() if 'model' in k.lower() or '.pt' in k.lower() or '.pth' in k.lower()]
        if model_artifacts:
            print(f"Found potential model artifacts: {model_artifacts}")
            # You could potentially register a model from these artifacts if needed
            
    # Strategy 4: Query for recent models across the project
    if not found_model:
        print("Strategy 4: Querying for recent models across the project...")
        models = Model.query_models(
            project_name="Guardian_Training",
            model_name=expected_model_name,
            only_published=False,  # Try both published and unpublished
            max_results=10,
            order_by=['-created'],  # Get the most recent models
        )
        
        if models:
            print(f"Found {len(models)} models via project query")
            found_model = models[0]  # Use the most recent model
            print(f"Selected most recent model: {found_model.id}")
    
    # Strategy 5: Check child tasks of the HPO task
    if not found_model:
        print("Strategy 5: Checking child tasks of the HPO process...")
        # Get child tasks
        child_tasks = Task.get_tasks(parent=hpo_task.id, status=['completed', 'stopped'])
        print(f"Found {len(child_tasks)} child tasks")
        
        for child_task in child_tasks:
            child_models = child_task.models.get('output', [])
            for model in child_models:
                if model.name == expected_model_name:
                    found_model = model
                    print(f"Found matching model {found_model.id} in child task {child_task.id}")
                    break
            if found_model:
                break
    
    # Strategy 6: Check the base task template
    if not found_model:
        print("Strategy 6: Checking models in the base task template...")
        base_task = Task.get_task(task_id=base_task_template_id)
        base_models = base_task.models.get('output', [])
        
        if base_models:
            print(f"Found {len(base_models)} models in base task {base_task_template_id}")
            for model in base_models:
                if model.name == expected_model_name:
                    found_model = model
                    print(f"Using fallback model from base task: {found_model.id}")
                    break
    
    # Final decision
    if not found_model:
        # In a production system, you could have a default fallback model
        print("WARNING: No suitable model found through any strategy")
        raise RuntimeError(
            f"Output model named '{expected_model_name}' not found through any search strategy. "
            "This could mean the training trial did not produce a publishable model."
        )
    
    # We have the clearml.Model object - now get its local path
    print(f"Found model {found_model.id}, retrieving local copy...")
    model_id = found_model.id
    
    try:
        local_model_path = found_model.get_local_copy()
        
        if not local_model_path or not os.path.exists(local_model_path):
            # Try an alternate approach
            print(f"Standard retrieval failed, trying alternate download approach...")
            local_model_path = os.path.join(output_dir, f"model_{model_id}.pt")
            found_model.download_model(local_model_path)
            
            if not os.path.exists(local_model_path):
                raise RuntimeError("Model download failed")
                
        print(f"Successfully retrieved model to: {local_model_path}")
    except Exception as e:
        print(f"Error retrieving model: {e}")
        raise RuntimeError(f"Failed to obtain a valid local copy for model ID {model_id}. Error: {str(e)}")
    
    print(f"Model retrieval successful - ID: {model_id}, Path: {local_model_path}")

    return best_hpo_trial_task.id, model_id, local_model_path


@PipelineDecorator.component(
    name="Evaluate_Model",
    return_values=["test_accuracy"],
    cache=False,
    packages=["torch", "scikit-learn", "numpy", "clearml"]
)
def evaluate_model(
    best_task_id: str,
    dataset_root_path: str,
    input_size: int = 34
):
    """
    Evaluates the best BiLSTM model on the test split.
    Returns test accuracy.
    """
    # Self-contained imports
    import sys, os
    import tempfile
    
    # Get the temp directory with our modules
    temp_dir = os.path.join(tempfile.gettempdir(), "guardian_modules")
    if os.path.exists(temp_dir) and temp_dir not in sys.path:
        sys.path.append(temp_dir)
        print(f"Added {temp_dir} to sys.path")
    
    print("DEBUG: sys.path =", sys.path)
    print("DEBUG: cwd =", os.getcwd())
    
    import torch
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from clearml import Task, Model
    
    # Import modules from the temp directory
    sys.path.insert(0, temp_dir)
    try:
        from pose_dataset import PoseDataset
        from model import ActionRecognitionBiLSTMWithAttention
        print("Successfully imported modules from temp directory")
    except ImportError as e:
        print(f"Error importing modules from {temp_dir}: {e}")
        # List files in the temp directory
        print(f"Files in {temp_dir}:", os.listdir(temp_dir) if os.path.exists(temp_dir) else "Directory not found")
        raise
    
    # Hardcode the action classes to avoid serialization issues
    action_classes = ["Falling", "No Action", "Waving"]
    
    # Create the output directory in temp folder if it doesn't exist
    output_dir = os.path.join(tempfile.gettempdir(), "clearml_guardian_output")
    os.makedirs(output_dir, exist_ok=True)
    output_uri = "file:///" + output_dir.replace("\\", "/")
    print(f"DEBUG: Evaluate Using output_uri: {output_uri}")

    # Initialize task with explicit output_uri
    task = Task.current_task()
    if task is None:
        task = Task.init(
            project_name="Guardian_Training",
            task_name="Evaluate_BiLSTM",
            output_uri=output_uri  # Explicitly set the output URI
        )
        print(f"DEBUG: Created new evaluation task with ID {task.id} and output_uri={output_uri}")
    
    # Get parameters from the best task
    bt = Task.get_task(task_id=best_task_id)
    params = bt.get_parameters_as_dict(flatten=True)
    hidden = int(params.get('General/hidden_size', 256))
    layers = int(params.get('General/num_layers', 4))
    dropout = float(params.get('General/dropout_rate', 0.1))

    # Get the model from the best task
    models = Model.query_models(
        project_name="Guardian_Training",
        model_name="BiLSTM_ActionRecognition",
        only_published=True,
        max_results=1
    )
    model_path = models[0].get_local_copy()

    # Load model
    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden,
        num_layers=layers,
        num_classes=len(action_classes),
        dropout_rate=dropout
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Prepare test data
    dataset = PoseDataset(dataset_root_path, action_classes)
    data, labels = dataset.data, dataset.labels
    _, test_data, _, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=42, stratify=labels
    )
    test_ds = PoseDataset(dataset_root_path, action_classes)
    test_ds.data, test_ds.labels = test_data, test_labels
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Compute accuracy
    preds, truths = [], []
    with torch.no_grad():
        for x, y in loader:
            out, _ = model(x)
            preds.extend(out.argmax(dim=1).tolist())
            truths.extend(y.tolist())
    acc = accuracy_score(truths, preds) * 100
    task.get_logger().report_scalar("Accuracy", "Test", acc, 0)
    return acc


@PipelineDecorator.pipeline(
    name="Guardian_Pipeline_BiLSTM",
    project="Guardian_Training"
)
def run_pipeline():
    # All action classes are hardcoded in each component to avoid serialization issues
    input_size = 34

    ds_id, ds_path = upload_pose_dataset()
    base_id, base_model = train_bilstm(
        dataset_root_path=ds_path,
        input_size=input_size
    )
    best_id, best_model_id, best_path = bilstm_hyperparam_optimizer(
        base_task_template_id=base_id,
        dataset_root_path=ds_path,
        input_size=input_size
    )
    evaluate_model(
        best_task_id=best_id,
        dataset_root_path=ds_path,
        input_size=input_size
    )


if __name__ == '__main__':
    # Set the config file path as the very first step
    if not load_clearml_conf("clearML/KongML.txt"):
        sys.exit(1)

    # Create and ensure the output directory exists
    # Use the system's temp directory which typically has appropriate permissions
    import tempfile
    # Create a unique subdirectory to avoid conflicts
    output_dir = os.path.join(tempfile.gettempdir(), "clearml_guardian_output")
    os.makedirs(output_dir, exist_ok=True)
    output_uri = "file:///" + output_dir.replace("\\", "/")
    print(f"DEBUG: Setting Task.__default_output_uri to: {output_uri}")
    print(f"DEBUG: Output directory is: {output_dir}")
    
    # Set the default output URI directly on the Task class
    # This is undocumented but appears to be what's used internally
    Task._Task__default_output_uri = output_uri

    # Now that CLEARML_CONFIG_FILE is set, ClearML operations should use it.
    PipelineDecorator.run_locally()
    run_pipeline()
