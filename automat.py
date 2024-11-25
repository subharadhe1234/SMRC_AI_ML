import os
import subprocess
import sys

import os
def main():
    current_dir = os.getcwd()
    requirements_path = os.path.join(current_dir, "requirements.txt")

    # Step 1: Install dependencies
    if os.path.exists(requirements_path):
        print("Installing dependencies from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True)
    else:
        print(f"{requirements_path} not found. Skipping installation.")

    # Step 2: Run scripts
    scripts = ["dataset.py", "main.py", "plotting.py"]
    scripts_dir = os.path.join(current_dir, "python code")

    for script in scripts:
        script_path = os.path.join(scripts_dir, script)
        print(f"Running {script_path}...")
        subprocess.run(["python", script_path])
        print("done")
        # run_script(script_path)
        
        

if __name__ == "__main__":
    main()
