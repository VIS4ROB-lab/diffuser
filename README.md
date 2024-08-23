# Diffuser

This repository contains the implementation of the multi-view semantic scene labeling pipeline described in:

Ruben Mascaro, Lucas Teixeira, and Margarita Chli. **Diffuser: Multi-View 2-D to 3-D Label Diffusion for Semantic Scene Segmentation**. IEEE International Conference on Robotics and Automation (ICRA), 2021.

Taking as input a 3D point cloud and a set of localized images processed by a 2D semantic segmentation network, *Diffuser* uses an efficient graphical model that leverages geometry to propagate class labels from the 2D image space to the 3D map. For more specific details, please check our **[[Paper]](https://www.research-collection.ethz.ch/handle/20.500.11850/484229)** and **[[Video]](https://youtu.be/WWqaFLgK5Kk)**.


## Environment Setup

This package requires Python 3.7+ and depends on the following libraries:
- [CuPy](https://cupy.dev/)
- [NumPy](https://numpy.org/)
- [Open3D](https://www.open3d.org/)
- [Pillow](https://python-pillow.org/)
- [SciPy](https://scipy.org/)

We recommend setting up a conda environment, as it simplifies the installation of the CUDA runtime libraries required by CuPy. A working conda environment can be created using the provided `environment-gpu.yml` file:

```shell
conda env create -f environment-gpu.yml
conda activate dfsr
```

**NOTE:** We also provide a CPU-only implementation that does not require CuPy and can be run on machines without a NVIDIA GPU. To set up a conda environment without support for GPU-accelerated computing, please use the `environment-cpu.yml` file:

```shell
conda env create -f environment-cpu.yml
conda activate dfsr
```


## Example Usage on ScanNet

### Dataset Preparation

Request access to the ScanNet data following the instructions in the [ScanNet repository](https://github.com/ScanNet/ScanNet/tree/master?tab=readme-ov-file#scannet-data). Upon acceptance of your request, you will be sent a link to a download script called `download-scannet.py`. Please copy this script into the `tools/datasets/` directory.

Then, clone the [ScanNet repository](https://github.com/ScanNet/ScanNet) and build the [ScanNet C++ Toolkit](https://github.com/ScanNet/ScanNet/tree/master?tab=readme-ov-file#scannet-c-toolkit) (this is required to extract the RGB-D frames and camera intrinsics and extrinsics from the compressed ScanNet `.sens` data):

```shell
git clone git@github.com:ScanNet/ScanNet.git
cd ScanNet/SensReader/c++/
make
```

Once the previous steps are completed, you can use the provided [download_scannet_scene.sh](tools/datasets/download_scannet_scene.sh) script to automatically download and preprocess any ScanNet scene. Example:

```shell
./tools/datasets/download_scannet_scene.sh scene0694_00 data/scannet ../ScanNet
```

### Running Diffuser

First, process the input frames with your semantic segmentation network of choice (e.g. [MSeg](https://github.com/mseg-dataset/mseg-semantic), [EMSANet](https://github.com/TUI-NICR/panoptic-mapping), etc.). 

Then, set up your own config file in the `config` directory, making sure that the input data paths point to the desired directories. Note that we provide an example config file ([scannet.ini](configs/scannet.ini)) that can be easily modified.

Finally, execute the [main.py](main.py) script as:

```shell
python main.py configs/scannet.ini
```


## Adding New Datasets

To test Diffuser on a custom dataset, please follow these steps:

1. Create a new file `diffuser/datasets/mydataset.py`.

```python
from .base_dataset import BaseLabelTaxonomy, BaseScene
from .builder import (register_dataset, register_label_mapping,
                      register_label_taxonomy)


# Adding a custom label taxonomy:
# Here you just need to define the class names and colors used for 3D semantic 
# labeling. The codebase assumes that the label of each class corresponds to 
# its position in the CLASSES array (e.g. 'class0' has label 0, class1' has 
# label 1, and so on).
@register_label_taxonomy('mylabeltaxonomy')
class MyLabelTaxonomy(BaseLabelTaxonomy):
  
  CLASSES = ('class0', 'class1', ...)
  PALETTE = ((r0, g0, b0), (r1, g1, b1), ...)

  def __init__(self):
        super().__init__()


# Adding a custom label mapping:
# Label mappings are needed if your semantic segmentation network of choice 
# uses a label taxonomy that does not exactly match the desired taxonomy for 3D 
# semantic labeling. The function should return a mapping from the input image 
# labels to the desired labels in 3D. In the example below, labels x and y in 
# image space would be mapped to label 0, while label z in image space would be
# mapped to label 1.  
@register_label_mapping('mylabelmapping')
def my_label_mapping():
  return [[x, y], [z], ...]


# Adding a custom dataset:
# To add a new dataset, you must provide the initialization function as well as
# the implementation of the abstract methods for data loading in the BaseScene 
# class. See `diffuser/datasets/scannet.py` for an example.
@register_dataset('scannet')
class MyDatasetScene(BaseScene):
  
  def __init__(self, arg1, arg2, ...):
        ...
```

2. Import the module in `diffuser/datasets/__init__.py`.

```python
from .mydataset import MyDatasetScene, MyLabelTaxonomy
```

3. Use the dataset by creating a new config file `configs/mydataset.ini`.

```python
[dataset]
dataset_name = mydataset
label_taxonomy = mylabeltaxonomy
img_labels_mapping = mylabelmapping
...
```


## License

This project is released under the [BSD 3-Clause License](LICENSE).


## Citation

If you use this code in your academic work, please consider citing:
```
@inproceedings{mascaro2021diffuser,
  title={Diffuser: Multi-View 2-D to 3-D Label Diffusion for Semantic Scene Segmentation},
  author={Mascaro, Ruben and Teixeira, Lucas and Chli, Margarita},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2021}
}
```