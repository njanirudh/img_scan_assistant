from setuptools import setup
from setuptools import setup, find_packages

setup(
    name='ImageScanAssistant',
    version='1.0',
    url='https://github.com/njanirudh/img_scan_assistant',
    description='Commandline application for cropping individual images from a large scanned image.',
    author='njanirudh',
    author_email='anijaya9@gmail.com',
    packages=find_packages(),
    install_requires=[
        'numpy',      # Example dependency
        'opencv-python'
    ],
    entry_points={
        'console_scripts': [
            'image_scan_cropper=main:main',  # Command-line interface if applicable
        ],
    },
)
