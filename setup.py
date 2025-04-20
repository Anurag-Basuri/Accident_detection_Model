from setuptools import setup, find_packages

setup(
    name="accident_detection",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "ultralytics",  # YOLOv8
        "deep-sort-realtime",
        "opencv-python",
        "numpy",
        "torch",
        "pyyaml",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Real-Time Accident Information System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 