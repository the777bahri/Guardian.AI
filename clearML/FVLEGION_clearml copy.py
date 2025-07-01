import os
import re
from clearml import Task
from clearml.automation.controller import PipelineDecorator
from pose_dataset import PoseDataset
import time
start_time = time.time()

def load_clearml_conf(conf_path):
    with open(conf_path, "r") as file:
        content = file.read()

    access_key = re.search(r'access_key"\s*=\s*"([^"]+)"', content).group(1)
    secret_key = re.search(r'secret_key"\s*=\s*"([^"]+)"', content).group(1)
    api_host = re.search(r'api_server:\s*(https?://[^\s]+)', content).group(1)
    web_host = re.search(r'web_server:\s*(https?://[^\s]+)', content).group(1)
    files_host = re.search(r'files_server:\s*(https?://[^\s]+)', content).group(1)

    os.environ["CLEARML_API_ACCESS_KEY"] = access_key
    os.environ["CLEARML_API_SECRET_KEY"] = secret_key
    os.environ["CLEARML_API_HOST"] = api_host
    os.environ["CLEARML_WEB_HOST"] = web_host
    os.environ["CLEARML_FILES_HOST"] = files_host


conf_path = os.path.join("Abdullah_improvments_with_logs", "FVLEGION.txt")
load_clearml_conf(conf_path)


@PipelineDecorator.component(return_values=["dataset_id", "dataset_path"])
def upload_pose_dataset():
    import os
    import logging
    from clearml import Dataset

    dataset_name = "Guardian_Dataset"
    dataset_project = "Guardian_Training"
    dataset_tags = ["version1"]
    dataset_path = r"../FVLegion_InitialDev-main/data collection v4/data"

    # Create a new dataset definition
    try:
        dataset = Dataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=dataset_tags
        )
    except Exception as e:
        logging.error(f"Failed to initiate dataset creation: {e}")
        return None, None

    logging.info("Checking for dataset changes...")

    try:
        prev_dataset = Dataset.get(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            only_completed=True
        )
        logging.info(f"Found previous dataset version: {prev_dataset.id}")
    except Exception as e:
        prev_dataset = None
        logging.warning(f"No previous dataset found: {e}")

    # Compare current and previous file sets
    current_files = set()
    for root, _, files in os.walk(dataset_path):
        for file in files:
            rel_path = os.path.relpath(os.path.join(root, file), dataset_path).replace("\\", "/")
            current_files.add(rel_path)

    if prev_dataset:
        try:
            prev_files = set(prev_dataset.list_files())
        except Exception as e:
            logging.error(f"Could not list previous dataset files: {e}")
            prev_files = set()

        if current_files == prev_files:
            logging.info("No changes detected in dataset. Reusing existing version.")
            try:
                dataset.delete(force=True)
            except Exception as e:
                logging.warning(f"Could not delete unused dataset object: {e}")
            return prev_dataset.id, dataset_path

    # Proceed to upload new version
    try:
        logging.info("Changes detected. Uploading new version.")
        dataset.add_files(path=dataset_path)
        dataset.upload()
        dataset.finalize()
        logging.info(f"Dataset uploaded and finalized: {dataset.id}")
        return dataset.id, dataset_path

    except Exception as e:
        logging.error(f"Upload failed: {e}")
        try:
            dataset.delete(force=True)
            logging.info("Deleted failed dataset version.")
        except Exception as cleanup_err:
            logging.warning(f"Cleanup failed: {cleanup_err}")
        return None, None

@PipelineDecorator.component()
def run_data_eda(dataset_path):
    import os
    import matplotlib.pyplot as plt

    def count_json_files(root_dir):
        json_count = 0
        for root, _, files in os.walk(root_dir):
            json_count += sum(1 for file in files if file.endswith('.json'))
        return json_count

    def plot_json_counts(root_dirs):
        json_counts = []
        labels = []

        for root_dir in root_dirs:
            count = count_json_files(root_dir)
            json_counts.append(count)
            labels.append(os.path.basename(root_dir))

        plt.figure(figsize=(10, 6))
        plt.bar(labels, json_counts, color='skyblue')
        plt.xlabel('Directory')
        plt.ylabel('Number of JSON Files')
        plt.title('Number of JSON Files per Action Class')
        for i, count in enumerate(json_counts):
            plt.text(i, count + 1, str(count), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("Abdullah_improvments_with_logs/eda/eda_json_counts.png")

    action_classes = ["Falling", "No Action", "Waving"]
    root_dirs = [os.path.join(dataset_path, action) for action in action_classes]
    plot_json_counts(root_dirs)

@PipelineDecorator.component(return_values=["train_loader", "val_loader", "test_loader", "input_size", "num_classes"])
def prepare_data(dataset_path):
    from torch.utils.data import DataLoader
    from sklearn.model_selection import train_test_split

    action_classes = ["Falling", "No Action", "Waving"]
    dataset = PoseDataset(dataset_path, action_classes)

    train_val_data, test_data, train_val_labels, test_labels = train_test_split(
        dataset.data, dataset.labels, test_size=0.2, random_state=42)
    print(f"Total samples: {len(dataset.data)}")

    train_data, val_data, train_labels, val_labels = train_test_split(
        train_val_data, train_val_labels, test_size=0.25, random_state=42)

    def make_loader(data, labels):
        d = PoseDataset(dataset_path, action_classes)
        d.data = data
        d.labels = labels
        return d

    train_loader = DataLoader(make_loader(train_data, train_labels), batch_size=32, shuffle=True)
    val_loader = DataLoader(make_loader(val_data, val_labels), batch_size=32, shuffle=False)
    test_loader = DataLoader(make_loader(test_data, test_labels), batch_size=32, shuffle=False)
    print(f"Train samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")

    return train_loader, val_loader, test_loader, 34, len(action_classes)

@PipelineDecorator.component(return_values=["model_artifact"])
def train_bilstm(train_loader, val_loader, input_size, hidden_size, num_layers, num_classes, learning_rate, weight_decay, num_epochs, dropout_rate, save_model=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from collections import defaultdict
    from clearml import Task
    from tqdm import tqdm  # Import tqdm for progress bar
    from model import ActionRecognitionBiLSTMWithAttention

    def get_device():
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    device = get_device()

    task = Task.current_task()
    task.set_parameter("General/learning_rate", learning_rate)
    task.set_parameter("General/weight_decay", weight_decay)
    task.set_parameter("General/num_epochs", num_epochs)
    task.set_parameter("General/hidden_size", hidden_size)
    task.set_parameter("General/num_layers", num_layers)
    task.set_parameter("General/dropout_rate", dropout_rate)
    print(f"learning_rate: {learning_rate}, weight_decay: {weight_decay}, num_epochs: {num_epochs}, hidden_size: {hidden_size}, num_layers: {num_layers}, dropout_rate: {dropout_rate}")

    model = ActionRecognitionBiLSTMWithAttention(input_size, hidden_size, num_layers, num_classes, dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop with tqdm progress bar
    for epoch in tqdm(range(num_epochs), desc="Training Progress", unit="epoch"):
        model.train()
        total_loss = 0
        correct, total = 0, 0
        class_attentions = defaultdict(list)

        # Batch progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch", leave=False) as batch_pbar:
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                out, attn = model(x)
                loss = criterion(out, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

                for i, label in enumerate(y.cpu().numpy()):
                    class_attentions[label].append(attn[i].detach().cpu().numpy())

                # Update batch progress bar
                batch_pbar.update(1)

        acc = correct / total if total > 0 else 0.0
        avg_loss = total_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out, _ = model(x)
                val_loss += criterion(out, y).item()
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader)

        # Console logging
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_loss:.4f} | Train Accuracy: {acc:.4f}")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Accuracy: {val_acc:.4f}")
        print(f"LR: {learning_rate} | WD: {weight_decay} | Layers: {num_layers} | Hidden: {hidden_size} | Dropout: {dropout_rate}")

        # ClearML logging
        logger = task.get_logger()
        logger.report_scalar("train", "train_loss", avg_loss, epoch)
        logger.report_scalar("train", "train_accuracy", acc, epoch)
        logger.report_scalar("validation", "validation_loss", avg_val_loss, epoch)
        logger.report_scalar("validation", "validation_accuracy", val_acc, epoch)
        logger.report_scalar("learning rate", "learning_rate", learning_rate, epoch)
        logger.report_scalar("weight decay", "weight_decay", weight_decay, epoch)
        logger.report_scalar("hidden size", "hidden_size", hidden_size, epoch)
        logger.report_scalar("num layers", "num_layers", num_layers, epoch)
        logger.report_scalar("dropout rate", "dropout_rate", dropout_rate, epoch)

        # Attention heatmap logging
        fig, axes = plt.subplots(1, len(class_attentions), figsize=(15, 3))
        for cls, attns in class_attentions.items():
            avg_attn = np.mean(attns, axis=0)
            ax = axes[cls] if len(class_attentions) > 1 else axes
            sns.heatmap([avg_attn], ax=ax, cmap='viridis', cbar=True)
            ax.set_title(["Falling", "No Action", "Waving"][cls])
            ax.set_xticks([]), ax.set_yticks([])

        plt.tight_layout()
        img_path = f"Abdullah_improvments_with_logs/training_outputs/attention_epoch_{epoch + 1}.png"
        plt.savefig(img_path)
        logger.report_image("Attention", f"Epoch {epoch + 1}", local_path=img_path)
        plt.close()

    # Save and upload model
    model_path = f"Abdullah_improvments_with_logs/models/bilstm_e{num_epochs}_l{num_layers}_h{hidden_size}.pth"

    torch.save(model.state_dict(), model_path)
    if save_model:
        best_model_path = f"Abdullah_improvments_with_logs/models/Best_bilstm_e{num_epochs}_l{num_layers}_h{hidden_size}.pth"
        torch.save(model.state_dict(), best_model_path)
        task.upload_artifact("best_trained_model", best_model_path)

    return task.id, "best_trained_model"

@PipelineDecorator.component(return_values=["best_params"])
def run_hpo_component(train_loader, val_loader, input_size, num_classes):
    import os
    import torch
    from clearml import Task
    from itertools import product
    from tqdm import tqdm
    from sklearn.metrics import accuracy_score
    from model import ActionRecognitionBiLSTMWithAttention

    # Device selection
    def get_device():
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    device = get_device()
    task = Task.current_task()
    logger = task.get_logger()

    # Define hyperparameter grid
    hidden_sizes    = [32, 64, 128, 256, 512]
    dropout_rates   = [0.2, 0.3, 0.5]
    learning_rates  = [0.01, 0.001, 0.0001, 0.00001]
    weight_decays   = [0.01, 0.001, 0.0001, 0.00001]
    num_epochs_list = [30, 50, 100, 150]
    num_layers_list = [1, 2, 3, 4,5]

    # Prepare combinations
    combos = list(product(
        hidden_sizes, dropout_rates, learning_rates,
        weight_decays, num_epochs_list, num_layers_list
    ))
    print(f"Total combinations: {len(combos)}")

    best_acc = 0.0
    best_config = {}

    # HPO trials loop
    with tqdm(total=len(combos), desc="HPO Trials", unit="trial") as pbar:
        for idx, (hs, dr, lr, wd, ne, nl) in enumerate(combos):
            trial_id = f"trial_{idx}_hs{hs}_dr{dr}_lr{lr}_wd{wd}_nl{nl}_ne{ne}"
            logger.report_text(f"Starting {trial_id}")

            # Log hyperparameter values
            logger.report_scalar("hyperparams", trial_id + "/hidden_size",    hs, idx)
            logger.report_scalar("hyperparams", trial_id + "/dropout_rate",   dr, idx)
            logger.report_scalar("hyperparams", trial_id + "/learning_rate",  lr, idx)
            logger.report_scalar("hyperparams", trial_id + "/weight_decay",   wd, idx)
            logger.report_scalar("hyperparams", trial_id + "/num_layers",     nl, idx)
            logger.report_scalar("hyperparams", trial_id + "/num_epochs",     ne, idx)

            # Build model and optimizer
            model = ActionRecognitionBiLSTMWithAttention(
                input_size=input_size,
                hidden_size=hs,
                num_layers=nl,
                num_classes=num_classes,
                dropout_rate=dr
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            criterion = torch.nn.CrossEntropyLoss()

            # Epoch-wise train and validate
            for epoch in range(ne):
                # Training
                model.train()
                t_loss, correct, total = 0.0, 0, 0
                for x, y in train_loader:
                    x, y = x.to(device), y.to(device)
                    out, _ = model(x)
                    loss = criterion(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    t_loss += loss.item()
                    correct += (out.argmax(1) == y).sum().item()
                    total += y.size(0)
                train_loss = t_loss / len(train_loader)
                train_acc  = correct / total if total else 0.0

                # Validation
                model.eval()
                v_loss, v_correct, v_total = 0.0, 0, 0
                preds, labels = [], []
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(device), y.to(device)
                        out, _ = model(x)
                        v_loss += criterion(out, y).item()
                        v_correct += (out.argmax(1) == y).sum().item()
                        v_total += y.size(0)
                        preds.extend(out.argmax(1).cpu().numpy())
                        labels.extend(y.cpu().numpy())
                val_loss = v_loss / len(val_loader)
                val_acc  = accuracy_score(labels, preds)

                # Report metrics under their respective charts
                logger.report_scalar("train_loss",          trial_id, train_loss, epoch)
                logger.report_scalar("train_accuracy",      trial_id, train_acc,  epoch)
                logger.report_scalar("validation_loss",     trial_id, val_loss,   epoch)
                logger.report_scalar("validation_accuracy", trial_id, val_acc,    epoch)

            # Update best config
            if val_acc > best_acc:
                best_acc = val_acc
                best_config = dict(
                    hidden_size=hs,
                    dropout_rate=dr,
                    learning_rate=lr,
                    weight_decay=wd,
                    num_layers=nl,
                    num_epochs=ne
                )
            pbar.update(1)

    # Save best config
    os.makedirs("Abdullah_improvments_with_logs", exist_ok=True)
    with open("Abdullah_improvments_with_logs/best_hyperparams.txt", "w") as f:
        for k, v in best_config.items():
            f.write(f"{k}: {v}\n")

    print(f"Best config: {best_config}, acc={best_acc:.4f}")
    return best_config


@PipelineDecorator.component()
def evaluate_model(model_task_id, model_artifact_name, test_loader, input_size, num_classes, hidden_size, num_layers, dropout_rate):
    import matplotlib.pyplot as plt
    from model import ActionRecognitionBiLSTMWithAttention
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix
    from clearml import Task
    import torch
    import joblib

    def visualize_spatial_saliency(model, input_sequence, predicted_class, frame_indices, skeleton_connections, raw_keypoints, save_dir):
       
        # Ensure gradients wrt input
        input_sequence = input_sequence.clone().detach().requires_grad_(True)

        # Forward pass
        logits, attn  = model(input_sequence)   
        score = logits[0, predicted_class]
        
        # Backward to get gradients
        model.zero_grad()
        score.backward()

        # saliency: absolute gradient of each feature
        grads = input_sequence.grad.data.abs().squeeze(0)  
        # collapse x/y per keypoint
        grads = grads.view(grads.size(0), 17, 2).mean(dim=2)  

        for t in frame_indices:
            kp_scores = grads[t].cpu().numpy()                
            kps = raw_keypoints[t]                            

            # normalize to [0,1] for plotting sizes/colors
            norm = (kp_scores - kp_scores.min()) / (kp_scores.ptp() + 1e-8)

            plt.figure(figsize=(4,4))
            for idx, (x,y) in enumerate(kps):
                plt.scatter(x, y, s=(norm[idx]*200)+10, c=norm[idx], cmap='hot')
            for i,j in skeleton_connections:
                x1,y1 = kps[i]; x2,y2 = kps[j]
                plt.plot([x1,x2], [y1,y2], 'k', linewidth=1, alpha=0.5)
            plt.title(f"Frame {t} class {predicted_class}")
            plt.axis("off")
            plt.savefig(f"{save_dir}/spatial_saliency_frame_{t}.png")
            plt.close()
    
    if os.path.exists("best_scaler.pkl"):
        scaler = joblib.load("best_scaler.pkl")
        print("[INFO] Loaded best scaler.")
    else:
        scaler = None

    def get_device():
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
        
    skeleton_connections = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
                            (0, 7), (7, 8), (8, 9), (7, 10), (10, 11), (11, 12),
                            (8, 13), (13, 14), (14, 15)]
    save_dir = "Abdullah_improvments_with_logs/saliency_outputs"
    device = get_device()

    model = ActionRecognitionBiLSTMWithAttention(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=dropout_rate
    ).to(device)

    if model_task_id and model_artifact_name:
        task = Task.get_task(model_task_id)
        model_path = task.artifacts[model_artifact_name].get_local_copy()
    else:
        model_path = "Abdullah_improvments_with_logs/models/best_model.pth" 

    model.load_state_dict(torch.load(model_path, weights_only=False))

    model.eval()

    y_true, y_pred = [], []

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        out, attn = model(x)
        preds = out.argmax(1)
        top_frames = attn.argmax(dim=1)
        
        for j in range(x.size(0)):
            raw_kps = x[j].detach().cpu().numpy().reshape(-1, 17, 2)
            model.train()  # needed for CuDNN backward
            with torch.enable_grad():  # <== Only here
                visualize_spatial_saliency(
                    model, x[j:j+1], preds[j].item(), [top_frames[j].item()],
                    skeleton_connections, raw_kps, save_dir
                )
            model.eval()

        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(1).cpu().numpy())


    labels = ["Falling", "No Action", "Waving"]
    report_str = classification_report(y_true, y_pred, target_names=labels)
    print(report_str)

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, xticklabels=labels, yticklabels=labels, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig("Abdullah_improvments_with_logs/evaluation/conf_matrix.png")

    Task.current_task().upload_artifact("confusion_matrix", "conf_matrix.png")

    with open("Abdullah_improvments_with_logs/evaluation/classification_report.txt", "w") as f:
        f.write(report_str)
    Task.current_task().upload_artifact("classification_report", "classification_report.txt")


@PipelineDecorator.pipeline(name="Guardian_Pipeline", project="Guardian_Training")
def run_pipeline():
    Task.current_task().add_requirements('requirements.txt')

    dataset_id, dataset_path = upload_pose_dataset()
    run_data_eda(dataset_path)
    train_loader, val_loader, test_loader, input_size, num_classes = prepare_data(dataset_path)

    best_params = run_hpo_component(train_loader, val_loader, input_size, num_classes)

    retrained_task_id, _ = train_bilstm(
        train_loader=train_loader,
        val_loader=val_loader,
        input_size=input_size,
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        num_classes=num_classes,
        learning_rate=float(best_params["learning_rate"]),
        weight_decay=float(best_params["weight_decay"]),
        num_epochs=int(best_params["num_epochs"]),
        dropout_rate=float(best_params["dropout_rate"]),
        save_model=True
    )

    _ = evaluate_model(
        model_task_id=retrained_task_id,
        model_artifact_name="best_trained_model",
        test_loader=test_loader,
        input_size=input_size,
        num_classes=num_classes,
        hidden_size=int(best_params["hidden_size"]),
        num_layers=int(best_params["num_layers"]),
        dropout_rate=float(best_params["dropout_rate"])
    )

    if os.path.exists("latest_train_task_id.txt"):
        os.remove("latest_train_task_id.txt")

if __name__ == "__main__":
    PipelineDecorator.run_locally()
    run_pipeline()
    finish_time = time.time()
    elapsed_time = (finish_time - start_time)/60
    print(f"Pipeline execution time: {elapsed_time:.2f} seconds")
    print("Pipeline completed successfully.")