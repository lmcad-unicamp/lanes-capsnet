# The Multi-lane CapsNet (MLCN)

We introduce Multi-Lane Capsule Networks (MLCN), which are a separable and resource efficient organization of Capsule Networks (CapsNet) that allows parallel processing while achieving high accuracy at reduced cost. A MLCN is composed of a number of (distinct) parallel lanes, each contributing to a dimension of the result, trained using the routing-by-agreement organization of CapsNet. Our results indicate similar accuracy with a much-reduced cost in number of parameters for the Fashion-MNIST and Cifar10 datsets. They also indicate that the MLCN outperforms the original CapsNet when using a proposed novel configuration for the lanes. MLCN also has faster training and inference times, being more than two-fold faster than the original CapsNet in the same accelerator. 

Full paper: https://arxiv.org/abs/1902.08431

# Source Code
This MLCN implementation used the @XifengGuo CapsNet Keras implementation (https://github.com/XifengGuo/CapsNet-Keras) as its base. All source code is available with the MIT license. 

# How to use it

Some of the details of how to install and use it can be found in the CapsNet-Keras project README. To support multi-lanes we introduced some new command line arguments:

- **--lane_size**  size of the lanes (an integer that should be greater or equal to one)
- **--lane_type**  type of the lanes (1 to mlcn1 and 2 for mlcn2)
- **--num_lanes**  number of lanes (an integer that should be greater or equal to 2)
- **--dataset**    mnist or cifar10 dataset
- **--dropout**    percentage of lanes being dropped out per batch

