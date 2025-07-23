import os
import subprocess

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    result = subprocess.run(["python", script_name], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(result.stdout)
    if result.stderr:
        print(f"Error in {script_name}:\n{result.stderr}")

def main():
    scripts = [
        "data/preprocess.py",     # Generates patches
        "train.py",               # Trains ViTSR (change inside to EDSR if needed)
        "test_vitsr.py",          # Evaluate ViTSR model
        "visualize.py",           # Save side-by-side LR / SR / HR comparisons
        "test_edsr.py",           # (Optional) Evaluate EDSR
    ]

    for script in scripts:
        run_script(script)

    print("\nAll scripts executed successfully.")

if __name__ == "__main__":
    main()
