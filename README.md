# CDGTN
The official code of paper, "Causal Diffused Graph-Transformer Network with Stacked Early Classification Loss for Efficient Stream Classification of Rumours".

## Abstract
The growth in social media has led to the increasing spread of unverified or false information. Automatically detecting rumours and accessing the veracity of rumours, i.e., false rumours, true rumours, or unverified rumours, are important and challenging tasks in social media analytics. This paper aims to build an effective and scalable stream classification framework for early fine-grained rumour classification, based on community response. We propose a Causal Diffused Graph- Transformer Network (CDGTN) to extract features from the source-reply graph in a social media conversation. Then, we propose source-guided incremental attention pooling to aggregate the encoded features with discrete timestamps. To improve the performance of early classification, we propose a stacked early classification loss, which aims to minimize the classification loss over the time instances. This can greatly improve the effectiveness of early classification of rumours. To improve the efficiency of streaming rumour verification, we propose a continued inference algorithm based on prefix-sum, which can greatly reduce the computational complexity of stream classification of rumours. Furthermore, we annotated the first Chinese rumour verification dataset, by extending the existing Chinese-Twitter dataset, namely CR-Twitter, originally for rumour detection. We conducted experiments on the Twitter15, Twitter 16, and the extended CR-Twitter datasets for finegrained rumour classification, to verify our proposed stream classification framework. Experimental results show that our proposed framework can significantly boost the effectiveness and efficiency of early stream classification of rumours.

## Datasets
The extended CR-Twitter can be downloaded below. Password is: CDGTN2023.

Extract the zip file and place the folder in preprocessed/CR_Twitter/

- [Extended CR-Twitter Dataset](https://connectpolyu-my.sharepoint.com/:u:/g/personal/15083269d_connect_polyu_hk/EdUFxIS-Ea9Igvk6ddku93wBxPYSehGT3OWAb3Y00J42Yw?e=BhWfgT)
## Model Checkpoints

The model checkpoints can be found in the releases [here](https://github.com/thcheung/CDGTN/releases/tag/checkpoint).

Download the checkpoint and place it into folder checkpoints/CDGTN_CR_Twitter_0_0/

## Citation

Please cite our work if you found this project useful.

```
@article{CHEUNG2023110807,
title = {Causal diffused graph-transformer network with stacked early classification loss for efficient stream classification of rumours},
journal = {Knowledge-Based Systems},
pages = {110807},
year = {2023},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2023.110807},
url = {https://www.sciencedirect.com/science/article/pii/S0950705123005579},
author = {Tsun-Hin Cheung and Kin-Man Lam},
keywords = {Neural network, Natural language processing, Social computing, Rumour classification, Early detection},
}
```
