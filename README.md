# DL-based Attenuation and Scatter Correction for 90Y SPECT Imaging

This repository provides deep learning models developed for **CT-free attenuation correction (AC)**, **Monte Carlo-based scatter correction (SC)**, and **combined attenuation and scatter correction (ASC)** in **90Y bremsstrahlung SPECT imaging**, using a Swin UNETR architecture. These models were trained on dose maps to improve **voxel-wise and organ-level dosimetry** for patients undergoing radioembolization (RE) or Selective Internal Radiation Therapy (SIRT).

🧠 **Read the refernce publication for more information:**:  

Mansouri Z., Salimi Y., Bianchetto Wolf N., Mainta I., Zaidi H.  
*CT-free attenuation and Monte-Carlo based scatter correction-guided quantitative 90Y-SPECT imaging for improved dose calculation using deep learning*  
_Eur J Nucl Med Mol Imaging_, 2025  
[DOI:10.1007/s00259-025-07191-5](https://doi.org/10.1007/s00259-025-07191-5)

## 🧪 Background

**90Y-SPECT** is widely used for post-therapy verification in liver cancer patients treated with radioembolization. However, its quantitative accuracy is compromised by photon attenuation and scatter, especially in CT-less settings. Traditional Monte Carlo methods are accurate but slow and hardware-dependent.

This work introduces fast and robust deep learning-based alternatives using dose-domain training on a relatively large patient dataset (n=190), leveraging a Swin UNETR model with transformer-based encoding.

## 🏥 Clinical Context
These models are beneficial in:

Standalone SPECT systems without CT

Hospitals lacking GPU access for MC simulations

Reducing processing time (from ~80 min to ~20 sec/GPU or ~6 min/CPU)

---
## Download Trained Models

https://drive.google.com/drive/folders/1FfzE_-_-mxiG6lyi4ap_JN6VlnThFfsd?usp=sharing

## 🧰 Repository Structure

To install this repository, simply run:

pip install git+https://github.com/ZahraMansouriMedPhys/DL-90Y-SPECT-Correction.git

We welcome any feedback, suggestions, or contributions to improve this project!

for any furtehr question please email me at: zahra.mansouri@unige.ch
