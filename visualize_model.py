import os
import torch
from torchsummary import summary
from torchviz import make_dot
from models.simple_cnn import SimpleCNN

from src.config_loader import CONFIG


def load_model(model_path, device):
    """Load the trained model."""
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def visualize_model(model, device, save_path):
    """Generate and save the model structure visualization."""
    dummy_input = torch.randn(1, 1, 28, 28, device=device)
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render(save_path, format="png")
    print(f"Model visualization saved at {save_path}.png")

# ==== Main Function ====

def main():

    # ==== Load Configuration ====

    DEVICE = CONFIG["device"]
    MODEL_PATH = os.path.join(CONFIG["output"]["models"], "best_model.pth")
    SAVE_PATH = os.path.join(CONFIG["output"]["plots"], "model_structure")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    model = load_model(MODEL_PATH, DEVICE)
    
    print("\nModel Summary:\n")
    summary(model, (1, 28, 28))
    
    visualize_model(model, DEVICE, SAVE_PATH)

# ==== Main Execution ====

if __name__ == "__main__":
    main()

