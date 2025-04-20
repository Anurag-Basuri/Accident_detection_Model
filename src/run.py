import asyncio
import streamlit.web.bootstrap
import streamlit.web.cli as stcli
import sys
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('torch').setLevel(logging.ERROR)

def main():
    # Get the absolute path to the app.py file
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Set up the event loop
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Set environment variables
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['PYTORCH_WARN_ONCE'] = '0'
    
    # Run Streamlit
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())

if __name__ == "__main__":
    main() 