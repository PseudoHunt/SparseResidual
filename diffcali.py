import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import timm

# Hyperparameters
BATCH_SIZE = 128
NUM_BITS = 3
INIT_A = 0.55  # Fixed power function exponent
FIXED_T = 100.5  # Fixed temperature for soft rounding
LR = 0.001  # Learning rate
EPOCHS = 10  # Training epochs for optimizing clipping range
NUM_ITERATIONS = 100  # Per-layer optimization iterations

# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load Pretrained ResNet18 from timm
import resnet18

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnet18.resnet18(pretrained=False, device=device)
model.to(device)

state_dict = torch.load('/content/resnet18.pt', map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)
model.eval()
model.to(device)

print("Accuracy BEFORE Quantization:")

# Function to evaluate accuracy
def evaluate(model, test_loader):
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Test Accuracy: {100 * correct / total:.2f}%")

# ====== Define Differentiable Quantization Model with Clipping Optimization ======
class ClippingOptimizedQuantization(nn.Module):
    def __init__(self, num_levels=2**NUM_BITS, fixed_a=INIT_A, fixed_T=FIXED_T):
        super().__init__()
        self.num_levels = num_levels
        self.fixed_a = fixed_a  # Fixed power exponent
        self.fixed_T = fixed_T  # Fixed temperature for soft rounding

        # Trainable min and max values for clipping
        self.w_min = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        self.w_max = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, w):
        # Apply power function transformation (fixed a)
        offset = w.mean()  # Centering the weights
        w_shifted = w - offset
        EPSILON = 1e-6
        w_transformed = torch.sign(w_shifted) * (torch.abs(w_shifted) ** self.fixed_a) + EPSILON

        # Ensure min and max clipping values are valid
        w_min_clamped = self.w_min.clamp(min=w_transformed.min().item(), max=w_transformed.max().item() - EPSILON)
        w_max_clamped = self.w_max.clamp(min=w_min_clamped.item() + EPSILON, max=w_transformed.max().item())

        # Normalize weights based on trainable clipping range
        w_normalized = (w_transformed - w_min_clamped) / (w_max_clamped - w_min_clamped + EPSILON)

        # Define quantization levels
        q_levels = torch.linspace(0, 1, self.num_levels, device=w.device)

        # Compute soft assignment weights (distance-aware rounding)
        distances = -torch.abs(w_normalized.unsqueeze(-1) - q_levels)  # Negative for softmax
        soft_weights = torch.softmax(distances * self.fixed_T, dim=-1)  # Fixed temperature

        # Compute soft quantized value as a weighted sum
        w_quantized = (soft_weights * q_levels).sum(dim=-1)

        # De-normalize back to original scale
        w_dequantized = w_quantized * (w_max_clamped - w_min_clamped) + w_min_clamped
        w_dequantized = (torch.abs(w_dequantized) ** (1/self.fixed_a)) * torch.sign(w_dequantized) + offset  # Descale

        return w_dequantized

# ====== Per-Layer Optimization Function ======
def optimize_per_layer(model, test_loader, num_iterations=NUM_ITERATIONS, lr=LR):
    model.to(device)
    model.eval()
    updated_state_dict = model.state_dict()
    quantization_layers = {}

    print("Starting per-layer quantization optimization...")

    # Get a batch of test images and labels before optimization
    data_iterator = iter(test_loader)
    images, labels = next(data_iterator)
    images, labels = images.to(device), labels.to(device)

    # Compute classification loss before optimization
    with torch.no_grad():
        outputs = model(images)
        pre_optimization_loss = nn.CrossEntropyLoss()(outputs, labels).item()
    print(f"Initial Classification Loss Before Optimization: {pre_optimization_loss:.6f}")

    for name, param in model.named_parameters():
        if "conv" in name and "weight" in name:
            print(f"Optimizing {name}...")
            layer_name = name.replace(".weight", "")

            # Initialize differentiable quantization module for this layer
            quantization_layers[layer_name] = ClippingOptimizedQuantization().to(device)
            quant_layer = quantization_layers[layer_name]

            optimizer = optim.Adam(quant_layer.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            original_weight = param.clone().detach()

            # Optimization loop
            prev_class_loss = 100
            for iter in range(num_iterations):
                optimizer.zero_grad()
                quantized_weight = quant_layer(original_weight)  # Apply quantization

                # Compute output difference
                quantized_output = nn.functional.conv2d(
                    images, quantized_weight, stride=param.shape[2], padding=param.shape[3]
                )
                original_output = nn.functional.conv2d(
                    images, original_weight, stride=param.shape[2], padding=param.shape[3]
                )

                # Compute losses
                reconstruction_loss = loss_fn(quantized_output, original_output)
                classification_loss = nn.CrossEntropyLoss()(model(images), labels)

                if prev_class_loss < classification_loss:
                    break
                prev_class_loss = classification_loss

                # Compute total loss
                final_loss = 0.1 * reconstruction_loss + 0.9 * classification_loss
                final_loss.backward()
                optimizer.step()

                if iter % 10 == 0:
                    print(f"Iter {iter}: recon_loss = {reconstruction_loss.item():.8f}, class_loss = {classification_loss.item():.4f}, min_w = {quant_layer.w_min.item():.4f}, max_w = {quant_layer.w_max.item():.4f}")

            updated_state_dict[name] = quant_layer(original_weight).detach()

    model.load_state_dict(updated_state_dict)
    print("Per-layer optimization complete.")

# Run forward pass to store activations
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        model(images)
        break  # Single batch needed

# Apply optimized quantization per layer
optimize_per_layer(model, test_loader)

# Evaluate Model After Quantization
evaluate(model, test_loader)
