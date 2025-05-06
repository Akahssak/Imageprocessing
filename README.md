# Low-Frequency Object Detection Pipeline

This project provides a pipeline for detecting low-frequency objects in images using image preprocessing, frequency analysis, segmentation, and evaluation. It includes a Streamlit app for interactive visualization and batch processing, as well as a simple CNN model for classification.

## Features

- Image preprocessing and augmentation
- Frequency domain filtering with circular and Gaussian low-pass filters
- Adaptive thresholding and morphological segmentation
- Evaluation metrics: precision, recall, F1-score, IoU
- Streamlit app with colorful UI for single image and batch processing
- Placeholder for CNN model training on segmented patches

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ipcv
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

To run the interactive Streamlit app locally:

```bash
streamlit run src/app.py
```

This will open the app in your default web browser.

## Deployment on Streamlit Cloud

You can deploy this app easily on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your code to a GitHub repository.

2. Log in to Streamlit Cloud and create a new app.

3. Connect your GitHub repository and select the branch and main file (`src/app.py`).

4. Specify any required secrets or environment variables if needed.

5. Deploy the app. Streamlit Cloud will install dependencies from `requirements.txt` automatically.

## Usage

- Use the sidebar to select between Single Image Processing, Batch Processing, Model Training, and About sections.

- Upload images or specify directories for batch processing.

- Adjust parameters such as resize dimensions, filter type, radius, and sigma.

- View processed images and evaluation metrics interactively.

## License

MIT License

## Contact

For questions or support, please contact [Your Name] at [your.email@example.com].
