# Facial Expression Recognition (TensorFlow/Keras)

Modernized facial expression recognition pipeline with clean Python modules, reproducible configs, and optional interactive notebooks.

> **Highlights**
> - Modular training and inference code in `facial_expression_recognition/`
> - YAML-based configuration for reproducibility
> - Ready-to-run CLI commands for training, image prediction, and webcam demos
> - Notebooks remain available under `notebooks/` for quick experimentation

---

## Project structure

```
.
├─ configs/                 # Example experiment configuration
├─ notebooks/               # Legacy exploration notebooks
├─ scripts/                 # Convenience runners for training and demos
└─ src/facial_expression_recognition/
   ├─ cli.py                # CLI entrypoint
   ├─ config.py             # Dataclass-driven configuration loader
   ├─ data.py               # Data generators and class resolution
   ├─ model.py              # CNN architecture definition
   ├─ training.py           # High-level training pipeline
   └─ inference.py          # Image + webcam inference utilities
```

## Quickstart

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Prepare your dataset**

   The loaders expect directory-based splits with one subfolder per class. Example:

   ```
   datasets/
     train/
       angry/...
       happy/...
       ...
     val/
       angry/...
       happy/...
       ...
   ```

   Update `configs/example_experiment.yaml` with your `train_dir` and `val_dir` paths. Adjust `class_names` if your labels differ.

3. **Train a model**

   ```bash
   python -m facial_expression_recognition.cli train --config configs/example_experiment.yaml
   ```

   Models and checkpoints are saved to `outputs/` by default.

4. **Predict a single image**

   ```bash
   python -m facial_expression_recognition.cli predict-image \
     --model outputs/model.h5 \
     --labels angry disgust fear happy neutral sad surprise \
     --image path/to/example.png
   ```

5. **Run the webcam demo**

   ```bash
   python -m facial_expression_recognition.cli predict-webcam \
     --model outputs/model.h5 \
     --labels angry disgust fear happy neutral sad surprise
   ```

   Press `q` to exit the live viewer.

## Configuration

All experiment settings live in YAML for transparency and reproducibility. Key sections:

- `dataset`: training/validation directories, image size, batch size, augmentation
- `model`: convolutional filter sizes, dense layers, dropout, learning rate, class count
- `training`: epochs, early stopping, reduce-on-plateau, and output paths

Copy `configs/example_experiment.yaml` to a new file and tune values per experiment.

## Notebook exploration

The original exploratory notebook is preserved in `notebooks/facial_expression3.ipynb`. Launch Jupyter or Colab to iterate quickly on ideas while keeping production-ready Python modules for deployment. 

## Useful scripts

- `scripts/train.py` – starts training with a config file (defaults to the example config)
- `scripts/webcam_demo.py` – lightweight wrapper around the webcam inference loop

Both scripts mirror the CLI behavior and can be customized for more advanced automation.

## Notes

- Default model expects grayscale 48×48 inputs. Adjust `color_mode` and `image_size` in the config if your dataset differs.
- For highest performance, ensure GPU-accelerated TensorFlow is installed in your environment.

