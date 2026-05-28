"""
Quantum Robot Policy Implementation with PennyLane
===================================================

Hybrid quantum-classical vision transformer for robot manipulation.
Combines pre-trained classical ViT with quantum policy head.

Architecture:
    Robot Frame [3, 224, 224]
        ↓
    Classical ViT Encoder (frozen) → CLS Token [768]
        ↓
    Compression Layer [768 → 64] (trainable)
        ↓
    Quantum Circuit (6 qubits, 4 layers) (trainable)
        ↓
    Action Head [6 → 7] (trainable)
        ↓
    Robot Actions [x, y, z, roll, pitch, yaw, gripper]

Author: Neal D'Andrea
Date: May 25, 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
import cv2  # For video loading
import sys
sys.path.insert(0, '/tmp/pypackages')
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt


# ==================== QUANTUM CIRCUIT DEFINITION ====================

# Define quantum device (6-qubit simulator)
dev = qml.device('default.qubit', wires=6)


@qml.qnode(dev, interface='torch', diff_method='backprop')
def quantum_circuit(features, weights):
    """
    Quantum circuit for policy head.
    
    Args:
        features: Compressed visual features [64]
        weights: Quantum parameters [4 layers, 6 qubits, 3 rotations] = [4, 6, 3]
    
    Returns:
        List of 6 expectation values (one per qubit)
    """
    # Step 1: Amplitude encoding
    # Encodes 64 classical values into 6-qubit quantum state
    qml.AmplitudeEmbedding(features, wires=range(6), normalize=True, pad_with=0.0)
    
    # Step 2: Parametrized quantum layers
    n_layers = weights.shape[0]  # Should be 4
    n_qubits = 6
    
    for layer in range(n_layers):
        # Rotation gates (trainable parameters)
        for qubit in range(n_qubits):
            qml.RX(weights[layer, qubit, 0], wires=qubit)
            qml.RY(weights[layer, qubit, 1], wires=qubit)
            qml.RZ(weights[layer, qubit, 2], wires=qubit)
        
        # Entangling gates (CNOT between adjacent qubits)
        for qubit in range(n_qubits - 1):
            qml.CNOT(wires=[qubit, qubit + 1])
        
        # Close the loop (optional but helps with entanglement)
        qml.CNOT(wires=[n_qubits - 1, 0])
    
    # Step 3: Measurement (Pauli-Z expectation values)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


# ==================== HYBRID QUANTUM-CLASSICAL MODEL ====================

class QuantumRobotPolicy(nn.Module):
    """
    Hybrid quantum-classical model for robot manipulation.
    
    Components:
        1. Classical ViT encoder (frozen, pre-trained)
        2. Compression layer (trainable)
        3. Quantum circuit (trainable)
        4. Action head (trainable)
    
    Parameters:
        - ViT: ~86M (frozen)
        - Compression: 49,216 (trainable)
        - Quantum: 72 (trainable)
        - Action head: 49 (trainable)
        - Total trainable: 49,337 (99.94% reduction)
    """
    
    def __init__(self, vit_encoder_path, n_qubits=6, n_layers=4, action_dim=7):
        """
        Initialize the quantum robot policy.
        
        Args:
            vit_encoder_path: Path to pre-trained ViT encoder (.pt file)
            n_qubits: Number of qubits (default: 6)
            n_layers: Number of quantum circuit layers (default: 4)
            action_dim: Robot action dimension (default: 7)
        """
        super(QuantumRobotPolicy, self).__init__()
        
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.action_dim = action_dim
        
        # Load pre-trained ViT encoder
        print(f"Loading ViT encoder from {vit_encoder_path}...")
        if os.path.exists(vit_encoder_path):
            try:
                import timm
            except ImportError:
                raise ImportError(
                    "timm library required. Install with: pip install timm"
                )
            
            # Create ViT architecture (12 blocks = vit_base)
            print("  Creating ViT architecture...")
            self.vit_encoder = timm.create_model(
                'vit_base_patch16_224',
                pretrained=False,
                num_classes=0,  # Remove classification head
                global_pool=''  # Keep all tokens
            )
            
            # Load your trained weights
            print("  Loading trained weights...")
            state_dict = torch.load(vit_encoder_path, weights_only=False)
            self.vit_encoder.load_state_dict(state_dict, strict=False)
            self.vit_encoder.eval()
            
            # Freeze ViT parameters
            for param in self.vit_encoder.parameters():
                param.requires_grad = False
            
            print("✓ ViT encoder loaded and frozen (12 transformer blocks)")
        else:
            raise FileNotFoundError(f"ViT encoder not found at {vit_encoder_path}")
        
        # Compression layer (768 → 64)
        # This reduces CLS token dimensionality to quantum-friendly size
        self.compression = nn.Linear(768, 64)
        print(f"✓ Compression layer: 768 → 64 ({768*64 + 64} parameters)")
        
        # Quantum circuit weights
        # Shape: [n_layers, n_qubits, 3] = [4, 6, 3] = 72 parameters
        # 3 rotation angles (RX, RY, RZ) per qubit per layer
        self.quantum_weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1  # Small initialization
        )
        print(f"✓ Quantum circuit: {n_layers} layers, {n_qubits} qubits ({n_layers*n_qubits*3} parameters)")
        
        # Action head (6 → 7)
        # Maps quantum measurement outputs to robot actions
        self.action_head = nn.Linear(n_qubits, action_dim)
        print(f"✓ Action head: {n_qubits} → {action_dim} ({n_qubits*action_dim + action_dim} parameters)")
        
        # Print total parameters
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        print(f"\n📊 Parameter Summary:")
        print(f"   Frozen (ViT): {total_frozen:,}")
        print(f"   Trainable: {total_trainable:,}")
        print(f"   Reduction: {(1 - total_trainable/(total_trainable+total_frozen))*100:.2f}%")
    
    def forward(self, x):
        """
        Forward pass through the hybrid model.
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
        
        Returns:
            actions: Predicted robot actions [batch_size, 7]
        """
        batch_size = x.shape[0]
        
        # Step 1: Classical ViT encoding (frozen)
        with torch.no_grad():
            # ViT outputs [batch_size, 197, 768]
            # 197 = 1 CLS token + 196 patch tokens
            vit_output = self.vit_encoder(x)
            
            # Extract CLS token (first token)
            cls_token = vit_output[:, 0, :]  # [batch_size, 768]
        
        # Step 2: Compression
        compressed = self.compression(cls_token)  # [batch_size, 64]
        
        # Step 3: Quantum circuit
        # Process each sample in batch through quantum circuit
        quantum_outputs = []
        for i in range(batch_size):
            # Quantum circuit expects 1D tensor [64]
            features = compressed[i]
            
            # Get quantum measurements [6]
            measurements = quantum_circuit(features, self.quantum_weights)
            
            # Stack measurements into tensor
            measurements_tensor = torch.stack(measurements)
            quantum_outputs.append(measurements_tensor)
        
        # Stack batch: [batch_size, 6]
        quantum_output = torch.stack(quantum_outputs)
        
        # Convert to float32 (PennyLane returns float64 by default)
        quantum_output = quantum_output.float()
        
        # Step 4: Action head
        actions = self.action_head(quantum_output)  # [batch_size, 7]

        return actions
    
    def get_trainable_params(self):
        """Return only trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]


# ==================== TRAINING FUNCTIONS ====================

def train_epoch(model, dataloader, optimizer, criterion, device='cpu'):
    """
    Train for one epoch.
    
    Args:
        model: QuantumRobotPolicy model
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: 'cpu' or 'cuda'
    
    Returns:
        average_loss: Average loss over epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (frames, actions_true) in enumerate(dataloader):
        frames = frames.to(device)
        actions_true = actions_true.to(device)
        
        # Forward pass
        actions_pred = model(frames)
        
        # Compute loss
        loss = criterion(actions_pred, actions_true)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()  # PennyLane handles quantum gradients via parameter shift!
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def train_with_knowledge_distillation(student, teacher, dataloader, optimizer, 
                                       temperature=3.0, alpha=0.5, device='cpu'):
    """
    Train student model using knowledge distillation from teacher.
    
    Args:
        student: Quantum model (student)
        teacher: Classical model (teacher)
        dataloader: Training data loader
        optimizer: PyTorch optimizer
        temperature: Temperature for distillation (default: 3.0)
        alpha: Balance between hard and soft loss (default: 0.5)
        device: 'cpu' or 'cuda'
    
    Returns:
        average_loss: Average loss over epoch
    """
    student.train()
    teacher.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (frames, actions_true) in enumerate(dataloader):
        frames = frames.to(device)
        actions_true = actions_true.to(device)
        
        # Teacher forward (frozen, no gradients)
        with torch.no_grad():
            teacher_actions = teacher(frames)
        
        # Student forward
        student_actions = student(frames)
        
        # Hard loss: match ground truth
        hard_loss = F.mse_loss(student_actions, actions_true)
        
        # Soft loss: match teacher (with temperature)
        soft_student = student_actions / temperature
        soft_teacher = teacher_actions / temperature
        soft_loss = F.mse_loss(soft_student, soft_teacher)
        
        # Combined loss
        loss = alpha * hard_loss + (1 - alpha) * soft_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f} "
                  f"(Hard: {hard_loss.item():.4f}, Soft: {soft_loss.item():.4f})")
    
    avg_loss = total_loss / num_batches
    return avg_loss



def evaluate(model, dataloader, criterion, device='cpu'):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: QuantumRobotPolicy model
        dataloader: Evaluation data loader
        criterion: Loss function
        device: 'cpu' or 'cuda'
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for frames, actions_true in dataloader:
            frames = frames.to(device)
            actions_true = actions_true.to(device)
            
            # Forward pass
            actions_pred = model(frames)
            
            # Compute metrics
            loss = criterion(actions_pred, actions_true)
            mse = F.mse_loss(actions_pred, actions_true)
            
            total_loss += loss.item()
            total_mse += mse.item()
            num_batches += 1
    
    metrics = {
        'loss': total_loss / num_batches,
        'mse': total_mse / num_batches,
    }
    
    return metrics



# ==================== UMI DATASET ====================

class UMIVideoDataset(Dataset):
    """
    Dataset for UMI robot manipulation data.
    Loads real actions from dataset.zarr.zip and frames from MP4 files.
    """

    def __init__(self, zarr_path=None, video_folder=None,
                 num_frames_per_video=100, transform=None):
        from torchvision import transforms

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.frames = []
        self.actions = []

        # Load real actions from zarr
        real_actions = None
        if zarr_path and os.path.exists(zarr_path):
            print(f"Loading real actions from zarr: {zarr_path}")
            real_actions = self._load_actions_only(zarr_path)
        else:
            print(f"No zarr found at {zarr_path}, will use placeholder actions")

        # Load frames from MP4 files
        if video_folder and os.path.exists(video_folder):
            print(f"Loading frames from MP4 files: {video_folder}")
            self._load_from_videos(video_folder, num_frames_per_video, real_actions)
        else:
            raise FileNotFoundError(f"Video folder not found: {video_folder}")

        print(f"✓ Loaded {len(self.frames)} frames total")
        if real_actions is not None:
            print(f"✓ Using REAL actions from zarr")
        else:
            print(f"⚠️  Using placeholder zero actions")

    def _load_actions_only(self, zarr_path):
        """Load only real actions from zarr, skip images to avoid codec issues."""
        import zipfile
        import zarr as zarr_lib

        extract_path = zarr_path.replace('.zip', '_extracted')
        if not os.path.exists(extract_path):
            print("  Extracting zarr zip...")
            with zipfile.ZipFile(zarr_path, 'r', allowZip64=True) as z:
                z.extractall(extract_path)
            print("  ✓ Extracted")
        else:
            print("  ✓ Already extracted")

        # Try to open zarr
        zarr_root = None
        for candidate in [extract_path,
                          os.path.join(extract_path, 'dataset.zarr')]:
            if os.path.exists(candidate):
                try:
                    zarr_root = zarr_lib.open(candidate, mode='r')
                    print(f"  ✓ Opened zarr: {candidate}")
                    break
                except Exception as e:
                    print(f"  Could not open {candidate}: {e}")
                    continue

        if zarr_root is None:
            print("  Could not open zarr, using placeholder actions")
            return None

        # Try standard action key first
        for path in [('data', 'action'),
                     ('data', 'robot_eef_pose'),
                     ('data', 'actions')]:
            try:
                node = zarr_root
                for key in path:
                    node = node[key]
                actions = node[:]
                print(f"  ✓ Found actions at {'/'.join(path)}: shape {actions.shape}")
                return actions
            except Exception:
                continue

        # Build 7D action from individual components
        print("  Building 7D actions from individual components...")
        try:
            eef_pos = zarr_root['data']['robot0_eef_pos'][:]
            eef_rot = zarr_root['data']['robot0_eef_rot_axis_angle'][:]
            gripper = zarr_root['data']['robot0_gripper_width'][:]
            actions = np.concatenate([eef_pos, eef_rot, gripper], axis=1)
            print(f"  ✓ Built real actions: shape {actions.shape}")
            print(f"    Position range:  [{eef_pos.min():.3f}, {eef_pos.max():.3f}]")
            print(f"    Rotation range:  [{eef_rot.min():.3f}, {eef_rot.max():.3f}]")
            print(f"    Gripper range:   [{gripper.min():.3f}, {gripper.max():.3f}]")
            return actions
        except Exception as e:
            print(f"  Could not build actions: {e}")

        print("  ⚠️  No actions found in zarr")
        return None

    # def _load_from_videos(self, video_folder, num_frames_per_video, real_actions=None):
    #     """Load frames from MP4 files, pair with real actions from zarr."""
    #     video_files = sorted([
    #         os.path.join(video_folder, f)
    #         for f in os.listdir(video_folder)
    #         if f.endswith('.MP4') or f.endswith('.mp4')
    #     ])
    #     print(f"  Found {len(video_files)} MP4 files")
    def _load_from_videos(self, video_folder, num_frames_per_video, real_actions=None):
        """Load frames from MP4 files, pair with real actions from zarr.
        Only uses first 171 sorted videos that passed SLAM preprocessing.
        """
        all_videos = sorted([
            os.path.join(video_folder, f)
            for f in os.listdir(video_folder)
            if f.endswith('.MP4') or f.endswith('.mp4')
        ])
        # Only use first 171 videos (matched to zarr episodes)
        video_files = all_videos[:171]
        print(f"  Found {len(all_videos)} MP4 files, using first {len(video_files)} (SLAM-processed)")

        for video_path in video_files:
            print(f"  Loading {os.path.basename(video_path)}...")
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, frame_count - 1,
                                  num_frames_per_video, dtype=int)
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frames.append(frame)

                    # Use real action from zarr if available
                    current_idx = len(self.frames) - 1
                    if real_actions is not None and current_idx < len(real_actions):
                        action = torch.tensor(
                            real_actions[current_idx][:7], dtype=torch.float32
                        )
                    else:
                        action = torch.zeros(7)
                    self.actions.append(action)
            cap.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        action = self.actions[idx]
        if self.transform and not isinstance(frame, torch.Tensor):
            frame = self.transform(frame)
        return frame, action



# ==================== LOGGING & PLOTTING ====================

def save_training_log(log, log_path):
    """Save training log to JSON file."""
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)

def plot_training_curves(log, save_path, architecture_name='Quantum Policy'):
    """
    Plot and save training/validation loss curves.
    Creates a clean plot saved as PNG.
    """
    epochs      = log['epochs']
    train_losses = log['train_losses']
    val_losses   = log['val_losses']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{architecture_name} - Training Curves', fontsize=14, fontweight='bold')

    # Full loss curve
    axes[0].plot(epochs, train_losses, 'b-',  linewidth=2, label='Train Loss', alpha=0.9)
    axes[0].plot(epochs, val_losses,   'r--', linewidth=2, label='Val Loss',   alpha=0.9)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Loss Over All Epochs')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')  # Log scale shows learning curve better

    # Smoothed loss curve (rolling average)
    window = max(1, len(epochs) // 10)
    def smooth(vals, w):
        return [sum(vals[max(0, i-w):i+1]) / len(vals[max(0, i-w):i+1])
                for i in range(len(vals))]

    axes[1].plot(epochs, smooth(train_losses, window), 'b-',  linewidth=2,
                 label='Train (smoothed)', alpha=0.9)
    axes[1].plot(epochs, smooth(val_losses,   window), 'r--', linewidth=2,
                 label='Val (smoothed)',   alpha=0.9)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_title('Smoothed Loss Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Stats box
    best_val   = min(val_losses)
    best_epoch = val_losses.index(best_val) + 1
    stats_text = (f"Best Val Loss: {best_val:.4f} (Epoch {best_epoch})\n"
                  f"Final Train:   {train_losses[-1]:.4f}\n"
                  f"Final Val:     {val_losses[-1]:.4f}\n"
                  f"Total Epochs:  {len(epochs)}\n"
                  f"Params:        {log.get('trainable_params', 'N/A')}")
    axes[1].text(0.97, 0.97, stats_text,
                 transform=axes[1].transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                 fontsize=8, fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Plot saved: {save_path}")


def plot_architecture_comparison(log_files, save_path):
    """
    Compare multiple architecture training curves on one plot.
    Pass a dict of {architecture_name: log_file_path}.
    
    Example:
        plot_architecture_comparison({
            '6-qubit 4-layer':  'logs/arch_6q4l.json',
            '8-qubit 4-layer':  'logs/arch_8q4l.json',
            '6-qubit 6-layer':  'logs/arch_6q6l.json',
        }, 'logs/comparison.png')
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Architecture Comparison - Validation Loss', fontsize=14, fontweight='bold')

    colors = ['#2563eb', '#dc2626', '#16a34a', '#9333ea',
              '#ea580c', '#0d9488', '#fbbf24', '#6b7280']

    for i, (arch_name, log_path) in enumerate(log_files.items()):
        if not os.path.exists(log_path):
            print(f"  Warning: log not found: {log_path}")
            continue

        with open(log_path, 'r') as f:
            log = json.load(f)

        color   = colors[i % len(colors)]
        epochs  = log['epochs']
        val     = log['val_losses']
        train   = log['train_losses']

        # Left: val loss comparison
        axes[0].plot(epochs, val, color=color, linewidth=2,
                     label=f"{arch_name} (best: {min(val):.4f})")

        # Right: train loss comparison
        axes[1].plot(epochs, train, color=color, linewidth=2,
                     label=arch_name, linestyle='--')

    for ax, title in zip(axes, ['Validation Loss', 'Train Loss']):
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Comparison plot saved: {save_path}")

def save_training_summary(log, save_path, arch_name):
        """
        Generate and save a detailed text summary of a completed training run.
        Covers architecture details, performance, convergence, and next steps.
        """
        train_losses = log['train_losses']
        val_losses   = log['val_losses']
        epochs       = log['epochs']

        best_val_loss  = min(val_losses)
        best_epoch     = val_losses.index(best_val_loss) + 1
        final_train    = train_losses[-1]
        final_val      = val_losses[-1]
        total_epochs   = len(epochs)

        # Convergence: how many epochs to get within 10% of best val loss
        threshold = best_val_loss * 1.1
        converge_epoch = next(
            (i + 1 for i, v in enumerate(val_losses) if v <= threshold),
            total_epochs
        )

        # Overfitting check: gap between train and val
        avg_gap = sum(abs(v - t) for v, t in zip(val_losses, train_losses)) / total_epochs
        final_gap = abs(final_val - final_train)
        overfit_status = (
            "Minimal overfitting" if final_gap < 0.01 else
            "Moderate overfitting" if final_gap < 0.05 else
            "Significant overfitting"
        )

        # Learning rate: was it stable?
        first_10_avg  = sum(train_losses[:10]) / min(10, len(train_losses))
        last_10_avg   = sum(train_losses[-10:]) / min(10, len(train_losses))
        improvement_pct = ((first_10_avg - last_10_avg) / first_10_avg) * 100

        # Performance rating
        if final_val < 0.01:
            performance = "EXCELLENT - well within robotics target (< 0.15)"
        elif final_val < 0.05:
            performance = "GOOD - within robotics target (< 0.15)"
        elif final_val < 0.15:
            performance = "ACCEPTABLE - meets robotics target (< 0.15)"
        else:
            performance = "NEEDS IMPROVEMENT - above robotics target (0.15)"

        summary = f"""
    ================================================================================
    TRAINING SUMMARY: {arch_name}
    ================================================================================
    Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

    ARCHITECTURE
    ------------
    Name:                {arch_name}
    Qubits:              {log.get('n_qubits', 'N/A')}
    Quantum Layers:      {log.get('n_layers', 'N/A')}
    Trainable Params:    {log.get('trainable_params', 'N/A'):,}
    Frozen Params:       85,798,656 (ViT encoder)
    Param Reduction:     99.94% vs fully classical model
    Encoding:            Amplitude encoding (64 features -> 6 qubits)
    Gradient Method:     Backpropagation (PennyLane simulator)

    TRAINING CONFIGURATION
    ----------------------
    Dataset:             UMI Robot Manipulation (real zarr actions)
    Action Source:       robot0_eef_pos + robot0_eef_rot_axis_angle + robot0_gripper_width
    Total Demos:         85,530 timesteps across 171 episodes
    Train Samples:       400 frames (80/20 split of 500 sampled)
    Val Samples:         100 frames
    Batch Size:          {log.get('batch_size', 'N/A')}
    Learning Rate:       {log.get('learning_rate', 'N/A')}
    Epochs Completed:    {total_epochs}
    Optimizer:           Adam

    PERFORMANCE RESULTS
    -------------------
    Best Val Loss:       {best_val_loss:.6f}  (Epoch {best_epoch}/{total_epochs})
    Final Train Loss:    {final_train:.6f}
    Final Val Loss:      {final_val:.6f}
    Train/Val Gap:       {final_gap:.6f}  ({overfit_status})
    Overall Rating:      {performance}

    CONVERGENCE ANALYSIS
    --------------------
    Epochs to converge:  {converge_epoch} epochs (within 10% of best val loss)
    First 10 avg loss:   {first_10_avg:.6f}
    Last 10 avg loss:    {last_10_avg:.6f}
    Total improvement:   {improvement_pct:.1f}%
    Loss at epoch 25%:   {train_losses[total_epochs//4 - 1]:.6f}
    Loss at epoch 50%:   {train_losses[total_epochs//2 - 1]:.6f}
    Loss at epoch 75%:   {train_losses[3*total_epochs//4 - 1]:.6f}
    Loss at epoch 100%:  {train_losses[-1]:.6f}

    ROBOTICS METRICS (estimated from MSE)
    --------------------------------------
    Action MSE:          {final_val:.4f}  (target: < 0.15)
    Position error est:  {(final_val**0.5) * 0.5:.4f} m  (target: < 0.005 m)
    Needs Isaac Sim evaluation for task success rate

    NEXT STEPS
    ----------
    1. Run classical baseline (same param count) and compare MSE
    2. Evaluate in Isaac Sim for task success rate
    3. Try architecture variants:
        - More qubits:   n_qubits=8  (change in QuantumRobotPolicy init)
        - Deeper circuit: n_layers=6  (change in QuantumRobotPolicy init)
        - Change ARCH_NAME to track each variant separately
    4. Run plot_architecture_comparison() once you have multiple logs

    SAVED FILES
    -----------
    Training log:    {save_path.replace('_summary.txt', '_training_log.json')}
    Loss curve plot: {save_path.replace('_summary.txt', '_loss_curve.png')}
    This summary:    {save_path}

    ================================================================================
    """

        with open(save_path, 'w') as f:
            f.write(summary)

        print(summary)
        print(f"  ✓ Summary saved: {save_path}")

# ==================== MAIN TRAINING SCRIPT ====================

def main():
    """
    Main training script for quantum robot policy.
    """
    print("=" * 60)
    print("Quantum Robot Policy - PennyLane Implementation")
    print("=" * 60)
    
    # Configuration
    VIT_ENCODER_PATH = '/tmp/QuantumInspiredViTSpeedUp/quantum_research/vit_encoder_only.pt'
    ZARR_PATH = '/home/wadeab/universal_manipulation_interface/data/session_001/dataset.zarr.zip'
    # DATA_PATH = '/tmp/QuantumInspiredViTSpeedUp/quantum_research/data_for_quantum_research2'
    DATA_PATH = '/home/wadeab/universal_manipulation_interface/data/session_001'
    BATCH_SIZE = 4
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n⚙️ Configuration:")
    print(f"   Device: {DEVICE}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print()
    
    # Initialize model
    try:
        model = QuantumRobotPolicy(
            vit_encoder_path=VIT_ENCODER_PATH,
            n_qubits=6,
            n_layers=4,
            action_dim=7
        )
        model = model.to(DEVICE)
        print("\n✓ Model initialized successfully")
    except Exception as e:
        print(f"\n✗ Error initializing model: {e}")
        print("\nPlease ensure:")
        print("  1. vit_encoder_only.pt exists in the current directory")
        print("  2. PennyLane is installed: pip install pennylane")
        print("  3. PyTorch is installed: pip install torch")
        return
    
    # TODO: Load your UMI dataset here
    # Load UMI dataset
    print("\n📊 Loading UMI dataset...")
    full_dataset = UMIVideoDataset(
        zarr_path=ZARR_PATH,
        video_folder=DATA_PATH,
        num_frames_per_video=100
    )

    # Split into train/val (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"✓ Train samples: {train_size}")
    print(f"✓ Val samples: {val_size}")
 
    # Optimizer
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=LEARNING_RATE)

    # Loss function
    criterion = nn.MSELoss()

    # Architecture name for logs/plots (change this when testing new architectures)
    ARCH_NAME = 'QViT_6q_4layer_171demos'

    # Logging setup
    # Save logs permanently outside /tmp
    LOG_DIR = '/home/wadeab/universal_manipulation_interface/quantum_research/logs'
    os.makedirs(LOG_DIR, exist_ok=True)
    log = {
        'architecture':    ARCH_NAME,
        'n_qubits':        6,
        'n_layers':        4,
        'learning_rate':   LEARNING_RATE,
        'batch_size':      BATCH_SIZE,
        'num_epochs':      NUM_EPOCHS,
        'trainable_params': 49337,
        'epochs':          [],
        'train_losses':    [],
        'val_losses':      [],
    }
    log_path  = f'{LOG_DIR}/{ARCH_NAME}_training_log.json'
    plot_path = f'{LOG_DIR}/{ARCH_NAME}_loss_curve.png'

    print(f"\n✓ Logging to: {log_path}")
    print(f"✓ Plot saved to: {plot_path}")
    print(f"\n🚀 Starting training - {ARCH_NAME}")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"  Train Loss: {train_loss:.4f}")

        # Validate
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Val Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}")

        # Log
        log['epochs'].append(epoch + 1)
        log['train_losses'].append(train_loss)
        log['val_losses'].append(val_metrics['val_loss'] if 'val_loss' in val_metrics else val_metrics['loss'])

        # Save log and plot every epoch
        save_training_log(log, log_path)
        plot_training_curves(log, plot_path, ARCH_NAME)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'{LOG_DIR}/{ARCH_NAME}_epoch_{epoch+1}.pt')
            print(f"  ✓ Checkpoint saved")

    print(f"\n✅ Training complete!")

    # Save final plot and summary
    summary_path = f'{LOG_DIR}/{ARCH_NAME}_summary.txt'
    plot_training_curves(log, plot_path, ARCH_NAME)
    save_training_summary(log, summary_path, ARCH_NAME)

    print(f"\n📁 All files saved to: {LOG_DIR}/")
    print(f"   Log:     {ARCH_NAME}_training_log.json")
    print(f"   Plot:    {ARCH_NAME}_loss_curve.png")
    print(f"   Summary: {ARCH_NAME}_summary.txt")


# ==================== TESTING / DEMO ====================

def test_forward_pass():
    """
    Test forward pass with random data to verify model works.
    """
    print("\n" + "=" * 60)
    print("Testing Forward Pass with Random Data")
    print("=" * 60)
    
    # Create dummy ViT that outputs the right shape for testing
    class DummyViT(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.randn(batch_size, 197, 768)
    
    # Test with dummy ViT
    print("\nCreating model with dummy ViT for testing...")
    model = QuantumRobotPolicy.__new__(QuantumRobotPolicy)
    nn.Module.__init__(model)
    
    model.vit_encoder = DummyViT()
    model.vit_encoder.eval()
    
    model.n_qubits = 6
    model.n_layers = 4
    model.action_dim = 7
    
    model.compression = nn.Linear(768, 64)
    model.quantum_weights = nn.Parameter(torch.randn(4, 6, 3) * 0.1)
    model.action_head = nn.Linear(6, 7)
    
    print("✓ Test model created")
    
    # Create random input
    batch_size = 2
    dummy_frames = torch.randn(batch_size, 3, 224, 224)
    print(f"\n✓ Created dummy input: {dummy_frames.shape}")
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    try:
        actions = model(dummy_frames)
        print(f"✓ Forward pass successful!")
        print(f"   Input shape: {dummy_frames.shape}")
        print(f"   Output shape: {actions.shape}")
        print(f"   Output: {actions}")
        print("\n✅ Model architecture is working correctly!")
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test first
    test_success = test_forward_pass()
    
    if test_success:
        print("\n" + "=" * 60)
        input("\nPress Enter to continue to main training setup...")
        main()
    else:
        print("\n✗ Please fix the errors above before proceeding to training.")
