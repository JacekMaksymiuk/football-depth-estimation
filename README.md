# Football Depth Estimation

## SoccerNet Challenge 2025

This repository contains the code for our submission to the [**SoccerNet Depth Estimation Challenge 2025**](https://www.soccer-net.org/tasks/monocular-depth-estimation).
Our method significantly outperforms the provided baselines, achieving **35–57% improvement** depending on the metric.
The accompanying paper is available [here](Player_Aware_Monocular_Depth_Estimation_for_Soccer_Broadcast_Frames.pdf).

### Results on SoccerNet-Depth Test Set

| Model                              | Abs Rel ×10⁻³ | RMSE ×10⁻³ | RMSE Log ×10⁻³ | Sq Rel ×10⁻⁴ | SILog |
|------------------------------------|---------------|------------|----------------|--------------|--------|
| ZoeDepth                           | 46.545        | 31.085     | 55.874         | 18.020       | 5.576  |
| DepthAnything                      | 4.105         | 3.680      | 6.130          | 0.262        | 0.613  |
| DepthAnything-ft-sn                | 2.584         | 2.401      | 4.167          | 0.125        | 0.417  |
| ZoeDepth-ft-sn                     | 2.429         | 2.343      | 4.002          | 0.121        | 0.400  |
| DepthAnything-ft (ours)            | 1.055         | 1.852      | 3.135          | 0.094        | 0.313  |
| **DepthAnything-player-ft (ours)** | **1.044**     | **1.510**  | **2.525**      | **0.055**    | **0.252** |


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

## Requirements

Model was trained on RTX4090 with 24GB of VRAM. It is possible to run it on 24GB of VRAM.
Minimum requirements is 24GB of VRAM, but it is possible to run it on 24GB of VRAM.

