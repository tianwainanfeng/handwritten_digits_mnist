import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")

def load_config(path=CONFIG_PATH):
    with open(path, "r") as file:
        return yaml.safe_load(file)

CONFIG = load_config()

# Example Usage
if __name__ == "__main__":
    print(CONFIG["data"]["raw"]["train_images"])
    print(CONFIG["data"]["raw"]["train_labels"])
    print(CONFIG["data"]["raw"]["test_images"])
    print(CONFIG["data"]["raw"]["test_labels"])

