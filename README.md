# Face Verification Project

This project implements a face verification system using deep learning techniques. It includes training a custom ResNet50 model for face classification and using the learned features for face verification.

## Project Structure

```
face_verification_project/
│
├── data/
│   ├── train/
│   ├── dev/
│   └── test/
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── validate.py
│   ├── test.py
│   └── utils.py
│
├── scripts/
│   └── run_experiment.py
│
├── requirements.txt
└── README.md
```

## Setup

1. Clone this repository:
   ```
   git clone https://github.com/realjules/image_classification.git
   cd image_classification
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Download the dataset:
   ```
   https://drive.google.com/file/d/1A2wFJtmvUtjwGgf8U-d3MeP8ePQvaRlb/view?usp=sharing
   ```

6. Update the paths in `src/config.py` to match your local setup.

## Running the Experiment

To run the full training and testing pipeline:

```
python scripts/run_experiment.py
```

This script will:
1. Initialize the model and datasets
2. Train the model for the specified number of epochs
3. Perform validation after each epoch
4. Save the best models based on classification and verification accuracy
5. Generate a submission file for the test set

## Monitoring

This project uses Weights & Biases (wandb) for experiment tracking. Make sure to set up your wandb account and log in before running the experiment.

## Additional Notes

- Adjust hyperparameters in `src/config.py` to optimize performance.
- For any issues or questions, please open an issue in the GitHub repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.