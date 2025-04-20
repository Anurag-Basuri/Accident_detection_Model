from setuptools import setup, find_packages

setup(
    name="accident-detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "streamlit==1.31.1",
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "Pillow==10.0.0",
        "matplotlib==3.7.2",
        "ultralytics==8.1.2",
        "torch==2.1.0",
        "torchvision==0.16.0",
        "tensorflow==2.15.0",
        "tensorflow-hub==0.14.0",
        "protobuf==3.20.3"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-Powered Real-Time Accident Information System",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8,<3.12",
    package_dir={"": "src"},
    include_package_data=True,
) 