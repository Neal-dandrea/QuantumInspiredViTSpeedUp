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
    Dataset for UMI robot manipulation videos.
    Loads frames from MP4 files and extracts actions.
    """
    
    def __init__(self, video_folder, num_frames_per_video=100, transform=None):
        """
        Args:
            video_folder: Path to folder containing MP4 files
            num_frames_per_video: Number of frames to sample from each video
            transform: Optional transform to apply to frames
        """
        import cv2
        from torchvision import transforms
        
        self.video_folder = video_folder
        self.num_frames_per_video = num_frames_per_video
        
        # Default transform for ViT (ImageNet normalization)
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
        
        # Find all MP4 files
        self.video_files = []
        for file in os.listdir(video_folder):
            if file.endswith('.MP4') or file.endswith('.mp4'):
                self.video_files.append(os.path.join(video_folder, file))
        
        print(f"Found {len(self.video_files)} video files:")
        for vf in self.video_files:
            print(f"  - {os.path.basename(vf)}")
        
        # Load all frames
        self.frames = []
        self.actions = []
        
        for video_path in self.video_files:
            print(f"Loading frames from {os.path.basename(video_path)}...")
            cap = cv2.VideoCapture(video_path)
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Sample frames evenly
            frame_indices = np.linspace(0, frame_count-1, num_frames_per_video, dtype=int)
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    # Convert BGR to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    self.frames.append(frame)
                    
                    # TODO: Replace with actual action labels from your dataset
                    # For now, using random actions as placeholder
                    # Actions: [x, y, z, roll, pitch, yaw, gripper]
                    random_action = torch.randn(7) * 0.1
                    self.actions.append(random_action)
            
            cap.release()
        
        print(f"✓ Loaded {len(self.frames)} frames total")
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        action = self.actions[idx]
        
        # Apply transform
        if self.transform:
            frame = self.transform(frame)
        
        return frame, action

# ==================== MAIN TRAINING SCRIPT ====================

def main():
    """
    Main training script for quantum robot policy.
    """
    print("=" * 60)
    print("Quantum Robot Policy - PennyLane Implementation")
    print("=" * 60)
    
    # Configuration
    VIT_ENCODER_PATH = r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\vit_encoder_only.pt'
    DATA_PATH = r'C:\Users\neald\Desktop\QuantumInspiredViTSpeedUp\quantum_research\data_for_quantum_research2'
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
        video_folder=DATA_PATH,
        num_frames_per_video=100  # 100 frames per video x 5 videos = 500 total
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

    print("\n⚠️  TODO: Load your UMI dataset")
    print("   Uncomment the dataset loading code above and provide your dataset class")
    print("   Expected format:")
    print("   - frames: [batch_size, 3, 224, 224]")
    print("   - actions: [batch_size, 7]")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.get_trainable_params(), lr=LEARNING_RATE)
    
    # Loss function
    criterion = nn.MSELoss()
    
    print("\n✓ Optimizer and loss function ready")
    print("\n🚀 Ready to train!")
    print("   Once you have your dataset loaded, uncomment the training loop below.")
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        print(f"  Val Loss: {val_metrics['loss']:.4f}, MSE: {val_metrics['mse']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'quantum_policy_epoch_{epoch+1}.pt')
            print(f"  ✓ Checkpoint saved")


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
