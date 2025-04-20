import os
import sys
import asyncio
import warnings
import logging
import streamlit.web.cli as stcli
from pathlib import Path

def main():
    """Main function to run the Streamlit application"""
    try:
        # Suppress warnings
        warnings.filterwarnings('ignore')
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.ERROR)
        logging.getLogger('ultralytics').setLevel(logging.ERROR)
        
        # Set event loop policy for Windows
        if os.name == 'nt':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # Get the absolute path to the app.py file
        current_dir = Path(__file__).parent
        app_path = current_dir / "app.py"
        
        # Set up command line arguments for Streamlit
        sys.argv = [
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
            "--server.address=localhost",
            "--server.port=8501",
            "--browser.serverAddress=localhost",
            "--browser.serverPort=8501",
            "--logger.level=error"
        ]
        
        # Run the Streamlit app
        sys.exit(stcli.main())
        
    except Exception as e:
        print(f"Error running the application: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 