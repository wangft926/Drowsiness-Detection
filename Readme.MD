# Drowsiness Detection using CNN and LSTM

A deep learning-based real-time driver drowsiness detection and alert system using CNN-LSTM architecture. The model analyzes eye movements, mouth openness (yawning), and head pose to accurately identify signs of fatigue, and triggers alert mechanisms such as buzzer sounds and LED display warnings.

## 🚗 Project Overview

Fatigue-induced drowsiness is a major contributor to road accidents. This project implements a non-intrusive solution that combines **Convolutional Neural Networks (CNN)** for spatial feature extraction and **Long Short-Term Memory (LSTM)** networks for temporal behavior modeling. It achieves a **97.83% accuracy** in classifying driver states: *Normal*, *Yawning*, and *Eye Closed*.

Key features:
- Real-time face landmark detection using MediaPipe
- EAR, MAR, and head pose feature extraction
- Temporal modeling with CNN-LSTM
- Hardware integration using Raspberry Pi, buzzer, and LED matrix
- Configurable inference and training pipeline

---

## 📁 Project Structure

```bash
├── config.ini                # Configuration file for training/inference
├── train.py                 # Script to train CNN-LSTM model
├── inference.py            # Real-time inference and alerting
├── dataset/                # Data loading and preprocessing scripts
├── model.py                # CNN-LSTM model definition
├── draw_landmarks.py       # Visualization utilities
├── utils/                  # Helper functions
├── README.md



To train on existing preprocessing,

In config.ini set use_existing_preprocessesd_data = True

Or to train on new dataset

In config.ini set use_existing_preprocessesd_data = False

Exceute python train.py

To run inference,

Execute python inference.py
"# Drowsiness-Detection-using-CNN-and-LSTM" 
