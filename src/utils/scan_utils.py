import torch

# ==== Scan Predictions ====

def scan_predictions(model, test_images, test_labels, device, digit, num_samples=5):
    """
    Scan test images to find correct and incorrect predictions for a specific digit.

    Args:
        model (torch.nn.Module): Trained model.
        test_images (torch.Tensor): Test images.
        test_labels (torch.Tensor): True labels.
        device (torch.device): CPU or CUDA.
        digit (int): The digit to scan.
        num_samples (int): Number of samples to collect.

    Returns:
        correct_samples (list): List of correctly classified images.
        incorrect_samples (list): List of incorrectly classified images.
    """
    model.eval()
    correct_samples = []
    incorrect_samples = []
    correct_labels, incorrect_labels = [], []
    correct_preds, incorrect_preds = [], []
    
    with torch.no_grad():
        for i in range(len(test_images)):
            img = test_images[i].unsqueeze(0).to(device)
            label = test_labels[i].item()
            pred = model(img).argmax(dim=1).item()

            if label == digit:
                if pred == digit and len(correct_samples) < num_samples:
                    correct_samples.append(test_images[i].cpu().numpy())
                    correct_labels.append(label)
                    correct_preds.append(pred)
                if pred != digit and len(incorrect_samples) < num_samples:
                    incorrect_samples.append(test_images[i].cpu().numpy())
                    incorrect_labels.append(label)
                    incorrect_preds.append(pred)

            if len(correct_samples) >= num_samples and len(incorrect_samples) >= num_samples:
                break

    return correct_samples, incorrect_samples, correct_labels, incorrect_labels, correct_preds, incorrect_preds
