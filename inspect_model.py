import torch
from cotracker.predictor import CoTrackerPredictor

def main():
    checkpoint = "checkpoints/scaled_online.pth"
    model = CoTrackerPredictor(checkpoint=checkpoint, offline=False, window_len=16)
    
    # Print modules that have 'drop' in their name
    count = 0
    for name, module in model.named_modules():
        if "drop" in name.lower():
            print(f"Found: {name} -> {module}")
            count += 1
    
    print(f"Total 'drop' related modules found: {count}")

if __name__ == "__main__":
    main()
