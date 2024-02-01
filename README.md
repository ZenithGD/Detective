# Detective

Detective: A tool for inferring camera pose and changes in the scene from a set of pictures
For Computer Vision assignment, MRGCV Zaragoza

## Installation

Install requirements:
```bash
pip install -r requirements.txt
```

Install LightGlue package locally:
```bash
python -m pip install -e ./detective/ext/LightGlue
```

Install CUDA-enabled PyTorch:
- Go to the [Torch installation page](https://pytorch.org/get-started/locally/) and select your distribution. 
    - In order to see what CUDA version you have installed:
    ```
    nvcc --version
    ```
    - Otherwise, install the appropriate CUDA version.
