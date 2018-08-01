# Instance Segmentation by Deep Coloring

[Paper preprint](https://arxiv.org/abs/1807.10007)


![Alt text](images/pipeline.png?raw=true "DeepColoring Pipeline")

### About
We propose a new and, arguably, a very simple reduction of instance segmentation to semantic segmentation. This reduction allows to train feed-forward non-recurrent deep instance segmentation systems in an end-to-end fashion using architectures that have been proposed for semantic segmentation. Our approach proceeds by introducing a fixed number of labels (colors) and then dynamically assigning object instances to those labels during training (coloring). A standard semantic segmentation objective is then used to train a network that can color previously unseen images. At test time, individual object instances can be recovered from the output of the trained convolutional network using simple connected component analysis. In the experimental validation, the coloring approach is shown to be capable of solving diverse instance segmentation tasks arising in autonomous driving (the Cityscapes benchmark), plant phenotyping (the CVPPP leaf segmentation challenge), and high-throughput microscopy image analysis.

### Test with pretrained models
1. For cvppp run notebook cvppp/cvppp.ipynb, follow code
2. For ecoli run notebook ecoli/ecoli.ipynb, follow code

### Trainging guidelines
1. [Guideline for cvppp](cvppp/README.md)
2. [Guideline for ecoli](ecoli/README.md)

### Instance segmentation for Cityscapes
(TF code to come soon)

