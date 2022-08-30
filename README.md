# Panoramas from Photons

### [Website](https://wisionlab.com/project/panoramas-from-photons/) | [Paper](https://wisionlab.com/wp-content/uploads/2023/08/Panoramas_from_Photons__ICCV_2023_fulltext.compressed.pdf) | [ICCP Demo](https://www.ubicept.com/blog/ubicept-at-iccp2023)

Official code repository for "Panoramas from Photons" (ICCV23), which enables high-quality single photon panoramas under fast motion and low-light. 

## Getting started
#### Create and Activate Virtual Env
We strongly recommend creating a virtual environment for this. Many tools exist, but here is an example using micromamba:

```shell
micromamba create --prefix=<path to env> python=3.10 -y
micromamba activate <path to env>
```

#### Clone Repo and Install Dependencies

```shell
git clone https://github.com/WISION-Lab/quantaodometry/
cd quantaodometry

pip install -r requirements.txt
pip install -e .
```

#### Running The Demo Code
Take a look at `notebooks/example_usage.ipynb` for a working example. 

## Citation

If you found this work useful, consider citing it:
```bibtex
@InProceedings{Jungerman_2023_ICCV,
    author = {Jungerman, Sacha and Ingle, Atul and Gupta, Mohit},
    title = {Panoramas from Photons},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2023},
}
```