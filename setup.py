from setuptools import setup, find_packages

setup(
    name="accident-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "opencv-python",
        "numpy",
        "Pillow",
        "matplotlib",
        "ultralytics"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Real-Time Accident Information System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 