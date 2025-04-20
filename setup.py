from setuptools import setup, find_packages

setup(
    name="accident-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.22.0",
        "opencv-python>=4.7.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "matplotlib>=3.5.0",
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Real-Time Accident Information System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    package_dir={"": "src"},
    include_package_data=True,
) 