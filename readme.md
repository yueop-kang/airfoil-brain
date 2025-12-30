# Airfoil Brain

**Airfoil Brain** is a data-driven framework for rapid prediction of airfoil lookup tables for arbitrary airfoil geometries.  
The framework consists of two core neural networks:

- **Airfoil Generation Network (AGN)**, which parameterizes a large and flexible airfoil design space  
- **Performance Prediction Network (PPN)**, which provides scalable and reliable aerodynamic predictions



## Environment

- Python 3.8.7  
- PyTorch 1.13.0 (CUDA 11.7)



## Installation

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
```

## Citation

If you use this code in your research, please cite the following publications:

```bibtex

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
```
