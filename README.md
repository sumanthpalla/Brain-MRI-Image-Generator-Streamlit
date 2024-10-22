# Brain MRI Generator

This Streamlit app generates Brain MRI images using a pre-trained diffusion model from Hugging Face. Users can upload an image, and the app will generate a corresponding Brain MRI image. The app also includes functionality to finetune the model on a custom dataset.

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/sumanthpalla/brain-mri-generator.git
   cd brain-mri-generator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the Streamlit app

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Upload an image using the file uploader.

4. Click the "Generate Brain MRI" button to create a Brain MRI image based on your input.

5. Download the generated image using the provided button.

### Finetuning the model

1. Prepare your dataset and update the `data_dir` in `config.yaml` to point to your dataset directory.

2. Run the training script:
   ```
   python train.py
   ```

3. The finetuned model will be saved in the `finetuned_model` directory.

4. To use the finetuned model in the Streamlit app, check the "Use finetuned model" box before generating images.

## Evaluation Metrics

The training script uses two main evaluation metrics:

1. Structural Similarity Index (SSIM): Measures the structural similarity between the generated and original images.
2. Peak Signal-to-Noise Ratio (PSNR): Measures the peak signal-to-noise ratio between the generated and original images.

These metrics are calculated during the validation phase of training and are printed after each epoch.

## Dataset

For finetuning, you should use a dataset of brain MRI images. One suitable public dataset is the [Brain Tumor MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection) from Kaggle. This dataset contains 253 brain MRI images, including both normal and tumor cases.

To use this dataset:
1. Download the dataset from Kaggle.
2. Extract the images to a directory.
3. Update the `data_dir` in `config.yaml` to point to this directory.


### More Information

1. A training script (`train.py`) that can finetune the model on a custom dataset.
2. Utility functions (`utils.py`) for calculating evaluation metrics (SSIM and PSNR).
3. A configuration file (`config.yaml`) for easy customization of training parameters.
4. Updates to the main app (`app.py`) to allow using the finetuned model.
5.Updated README with instructions for finetuning and information about the dataset and evaluation metrics.

The chosen public dataset for this project is the Brain Tumor MRI Dataset from Kaggle, which contains a good variety of brain MRI images for finetuning purposes.
The evaluation metrics (SSIM and PSNR) are calculated during the validation phase of training and provide a quantitative measure of the generated image quality compared to the original images.


To use this updated project:

Set up the environment and install dependencies as described in the README.
Prepare your dataset and update the config.yaml file.
Run the training script to finetune the model.
Use the Streamlit app to generate Brain MRI images, with the option to use the finetuned model.


## Note

This app is for demonstration purposes only. The generated images should not be used for medical diagnosis or treatment.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
