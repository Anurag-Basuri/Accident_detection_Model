from setuptools import setup, find_packages

setup(
    name="accident-detection",
    version="0.2",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.31.0",
        "opencv-python>=4.9.0",
        "numpy>=1.26.0",
        "Pillow>=10.2.0",
        "matplotlib>=3.8.0",
        "ultralytics>=8.1.0",
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "tensorflow>=2.12.0",
        "pyyaml>=6.0.0",
        "scikit-learn>=1.2.0",
        "tqdm>=4.65.0",
        "seaborn>=0.12.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Real-Time Accident Information System",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    include_package_data=True,
)