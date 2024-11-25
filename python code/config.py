import os
from pathlib import Path
from dotenv import load_dotenv, set_key

def setup_project_env():
    """
    Sets up the PROJECT_ROOT in a .env file dynamically.
    """
    # Resolve the project root dynamically (two levels up from this file)
    project_root = Path(__file__).resolve().parents[1]
    
    # Path to the .env file
    env_path = project_root / ".env"
    
    # Check if the .env file exists
    if not env_path.exists():
        # Create and write the PROJECT_ROOT variable
        with open(env_path, "w") as env_file:
            env_file.write(f"PROJECT_ROOT={project_root}\n")
        print(f"Created .env file with PROJECT_ROOT={project_root}")
    else:
        # Load existing .env file and check if PROJECT_ROOT is set
        load_dotenv(dotenv_path=env_path)
        current_root = os.getenv("PROJECT_ROOT")
        if current_root != str(project_root):
            # Update the PROJECT_ROOT in .env
            set_key(env_path, "PROJECT_ROOT", str(project_root))
            print(f"Updated PROJECT_ROOT in .env to {project_root}")
    
    # Load the updated environment variable into the session
    load_dotenv(dotenv_path=env_path)

    # Return the project root as a Path object
    return project_root

# Call setup_project_env and make project_root globally accessible
project_root = setup_project_env()

# Example usage
if __name__ == "__main__":
    print(f"Project Root: {project_root}")
