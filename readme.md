# Airfoil Brain
Airfoil Brain is data-driven framework for rapid prediction of C81 tables of arbitrary airfoil geometries. The framework consists of two core NNs: the airfoil generation network (AGN), which parameterizes a large airfoil design space, and the performance prediction network (PPN), which provides scalable and reliable predictions.

# Version
python 3.8.7
pytorch 1.13.0+cu117

# Installation
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117

# Citation
If you use this code in your research, please cite the following publications:

@article{kang2025intuitive,
  title   = {Intuitive and feasible geometric representation of airfoil using variational autoencoder},
  author  = {Kang, Yu-Eop and Lee, Dawoon and Yee, Kwanjung},
  journal = {Journal of Computational Design and Engineering},
  volume  = {12},
  number  = {2},
  pages   = {27--48},
  year    = {2025},
  publisher = {Oxford University Press}
}

@article{kang2025airfoilbrain,
  title   = {<to be updated>},
  author  = {Kang, Yu-Eop and others},
  journal = {Structural and Multidisciplinary Optimization},
  year    = {2025},
  note    = {to appear}
}
