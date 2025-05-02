import subprocess
import sys

def install_spacy_model():
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("Successfully installed en_core_web_sm")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install en_core_web_sm: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_spacy_model()