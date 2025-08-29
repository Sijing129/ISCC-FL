# Integrated Sensing, Communication, and Computation for Over-the-Air Federated Edge Learning

This repository contains the official implementation of our paper "Integrated Sensing, Communication, and Computation for Over-the-Air Federated Edge Learning".

## Framework Overview

Our proposed framework is illustrated below:
 ![Framework](ISCC_FL_System_Model.pdf)
*Figure 1: An ISCC-based Air-FEEL system.*

## How to use it
Following are the brief introductions of each file.
* main_fed.py is the main file.
* data_set.py is the pre-processor of the dataset.
* Folder data and model include the human motion recognition datasets, and the AI models, respectively.

Notice that for the optimization algorithm about the network resources, this version just provides an example value.

## Citation

If you find our work useful for your research and projects, please consider citing our paper and starring our project!

```bibtex
@ARTICLE{11143883,
  author={Wen, Dingzhu and Xie, Sijing and Cao, Xiaowen and Cui, Yuanhao and Xu, Jie and Shi, Yuanming and Cui, Shuguang},
  journal={IEEE Transactions on Wireless Communications}, 
  title={Integrated Sensing, Communication, and Computation for Over-the-Air Federated Edge Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Sensors;Convergence;Atmospheric modeling;Computational modeling;Resource management;Training;Data models;Servers;Artificial intelligence;Wireless sensor networks;Over-the-air federated edge learning;sensing-communication-computation integration;convergence analysis;resource allocation},
  doi={10.1109/TWC.2025.3598997}
}
