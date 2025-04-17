## Requirements

Model was trained on RTX4090 with 24GB of VRAM. It is possible to run it on 24GB of VRAM.
Minimum requirements is 24GB of VRAM, but it is possible to run it on 24GB of VRAM.

## Usage

1. Clone the repository:
    ```bash
    git clone https://github.com/JacekMaksymiuk/football-depth-estimation.git
    ```

2. Move to the cloned directory:
    ```bash
    cd football-depth-estimation
    ```
   
3. Install the required packages (Python 3.10) (preferably in a virtual environment):
    ```bash
    pip install -r requirements.txt
    ```
   
4. Example usage. Pipeline is using pretrained "da_ft_epoch24.pth" and "player_ft.pth", which are downloaded automatically from HuggingFace. In next section there is notebook for reproducing training of these models.
    ```python
    from depth_estimator import DepthEstimator
    
    de = DepthEstimator()
    
    de.predict(
        img_path='/path/to/color/img.png',
        output_path='/path/to/output/depth.png',
    )
    ```
   
## Training (for reproducibility)

1. Prepare Jupiter Lab environment (preferably in a virtual environment):
    ```bash
    pip install notebook jupyterlab ipywidgets
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    ```
   
2. Run models trainings and fine-tuning, by running the [train.ipynb](train.ipynb) notebook. 
**Important: set output directory to the one you want to save all files to.**
You'll need 200GB of free disk space (although less than 100GB will be taken up after the training is complete).

3. At the end models will be saved in the output directory and also results for test set will be saved in the output directory.
