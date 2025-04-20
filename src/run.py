import os
import sys
import asyncio
import warnings
import logging
from streamlit.web import cli as stcli
from streamlit.web import bootstrap

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def main():
    # Set event loop policy for Windows
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Construct the absolute path to app.py
    app_path = os.path.join(project_root, 'src', 'app.py')
    
    # Run the Streamlit app
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.headless",
        "true",
        "--server.runOnSave",
        "false",
        "--browser.serverAddress",
        "localhost",
        "--browser.serverPort",
        "8501"
    ]
    
    try:
        sys.exit(stcli.main())
    except Exception as e:
        print(f"Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 