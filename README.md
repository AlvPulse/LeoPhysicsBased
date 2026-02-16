# Harmonic Event Detector

This project detects harmonic events in audio signals using three methods:
1.  **Baseline Heuristic**: Simple peak detection logic.
2.  **Linear Model**: A learned linear combination of harmonic quality metrics (SNR, Power).
3.  **Graph Neural Network (GNN)**: A GNN trained on the harmonic structure graph.

## Setup

1.  **Generate Data**:
    Run the data generation script to create synthetic training and testing data in `data/yes` and `data/no`.
    ```bash
    python generate_data.py
    ```

2.  **Train Models**:
    Train the Linear and GNN models on the generated data.
    ```bash
    python train.py
    ```
    This will save model weights to `models/`.

3.  **Run Inference (Headless)**:
    Evaluate the models on all data files and generate a confusion matrix.
    ```bash
    python main_headless.py
    ```

4.  **Visualize Results**:
    Visualize the detection process (STFT, Tracking, Probability) for a single file.
    ```bash
    python main.py data/yes/sample_001.wav
    ```

## Structure

*   `src/`: Core logic modules.
    *   `config.py`: Hyperparameters.
    *   `signal_processing.py`: STFT and peak finding.
    *   `harmonic_detection.py`: Iterative harmonic tracking logic.
    *   `feature_extraction.py`: Feature vector and graph construction.
    *   `models.py`: PyTorch model definitions.
    *   `dataset.py`: PyTorch Dataset class.
*   `train.py`: Training script.
*   `main.py`: Visualization script.
*   `main_headless.py`: Batch evaluation script.
