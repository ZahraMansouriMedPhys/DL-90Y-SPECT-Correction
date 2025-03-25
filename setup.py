from setuptools import setup, find_packages

setup(
    name="DL-90Y-SPECT-Correction",
    #version="0.1.0",
    author="Zahra Mansouri",
    author_email="Zahra.mansouri@unige.ch",
    description="DL models for CT-free attenuation and monte-carlo based scatter corrections for 90Y-SPECT images",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ZahraMansouriMedPhys/DL-90Y-SPECT-Correction",
    packages=find_packages(),
    py_modules=["DL_inference", "image_utils"],
    install_requires=[
        "pandas",
        "tqdm",
        "termcolor",
        "glob2", 
        "torch",
        "SimpleITK",
        "multiprocess",
        "numpy", 
        "natsort", 
        "monai", 
        "nibabel", 
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)