# Measuring Representational Shifts in Continual Learning: A Linear Transformation Perspective

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2505.20970-B31B1B.svg)](https://arxiv.org/abs/2505.20970)
[![Conference](https://img.shields.io/badge/ICML-2025-1A73E8.svg)](https://icml.cc/)

Official implementation of the paper  
**"Measuring Representational Shifts in Continual Learning: A Linear Transformation Perspective"**  
by *Joonkyu Kim, Yejin Kim, and Jy-yong Sohn*  
accepted at **ICML 2025**.

---

## 🧠 Overview

This repository provides the official codebase for our ICML 2025 paper, which introduces a **linear transformation framework** for quantifying representational shifts in continual learning models.  
The framework allows researchers to interpret how internal feature spaces evolve across learning stages, with theoretical and empirical analyses on benchmark datasets.

---

## 📦 Installation

Clone the repository and install all dependencies as follows:

```bash
pip install black==21.6b0 \
            aiohttp==3.7.4 \
            aiohttp_cors==0.7.0 \
            numpy==1.20.3 \
            prettytable==2.1.0 \
            matplotlib==3.4.2 \
            tensorboard==2.5.0 \
            scipy==1.7.0 \
            Pillow==8.2.0 \
            jupyter==1.0.0 \
            poethepoet==0.10.0 \
            pandas==1.3.3 \
            tqdm==4.62.3
````

All experiments were tested with **Python 3.8+** and **CUDA 11.3**.


---

## 📝 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{kim2025measuring,
  title={Measuring Representational Shifts in Continual Learning: A Linear Transformation Perspective},
  author={Kim, Joonkyu and Kim, Yejin and Sohn, Jy-yong},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
```

