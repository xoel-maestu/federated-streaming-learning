import os
import re
import subprocess
import time

MODELS = [
    ("mlp", 50),
    ("cnn", 25),
    ("mobilenet", 10),
    ("resnet", 5),
]

CONFIG_FILE = "pyproject.toml"

def update_config(model, rounds):
    """Update model-type and num-server-rounds in pyproject.toml."""
    
    with open(CONFIG_FILE, "r") as f:
        content = f.read()

    # Replace model type
    content = re.sub(
        r'model-type\s*=\s*".*"', 
        f'model-type = "{model}"', 
        content
    )

    # Replace num rounds
    content = re.sub(
        r'num-server-rounds\s*=\s*\d+', 
        f'num-server-rounds = {rounds}', 
        content
    )

    with open(CONFIG_FILE, "w") as f:
        f.write(content)

    print(f"> Updated config: model={model}, rounds={rounds}")

def run_experiment(model):
    """Run Flower for the current configuration and save output to a log file."""

    print(f"\n[{time.strftime('%H:%M:%S')}] Running experiment for {model}...")

    # Create logs directory if needed
    os.makedirs("logs", exist_ok=True)

    log_path = f"logs/{model}.log"

    # Open the log file in write mode
    with open(log_path, "w") as log_file:
        # Start the Flower process
        process = subprocess.Popen(
            ["flwr", "run", "."],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            universal_newlines=True
        )

        # Stream output both to console and log file
        for line in process.stdout:
            print(line, end="")     # Show on console
            log_file.write(line)    # Save to log

        process.wait()

        if process.returncode == 0:
            print(f"[{time.strftime('%H:%M:%S')}] Experiment OK. Log saved to {log_path}")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR: exit code {process.returncode}")
            print(f"Log saved to {log_path}")
            print("Continuing to next experiment...")

def main():
    print("--- Starting experiments ---")
    print(f"Models: {[m for m,_ in MODELS]}")
    
    for model, rounds in MODELS:
        update_config(model, rounds)
        time.sleep(1)
        
        run_experiment(model)
        time.sleep(5)

    print("\n--- All experiments completed ---")

if __name__ == "__main__":
    if not os.path.exists(CONFIG_FILE):
        print(f"ERROR: {CONFIG_FILE} not found")
    else:
        main()
