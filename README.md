# PLR(AAAI2023)

This is the MindSpore implementation of PLR in the following paper.

AAAI 2023: Partial-Label Regression

# [PLR Description](#contents)

This paper provides the first attempt to investigate partial-label regression, where each training example is annotated with a set of real-valued candidate labels.

1) Propose a simple baseline method that takes the average loss incurred by candidate labels as the predictive loss to be minimized for model training.

2) Propose an identification method that takes the least loss incurred by candidate labels as the predictive loss to be minimized for model training.

3) Propose a progressive identification method that differentiates candidate labels by associating their incurred losses with progressively updated weights.

4) Show that the identification method and the progressive identification method are model-consistent, which indicates that the learned model converges to the optimal model.

# [Dataset](#contents)

Our experiments are conducted on seven widely used benchmark regression datasets to test the performance of our PLR, which are Abalone, Airfoil, Auto-mpg, Housing, Concrete, Power-plant and Cpu-act. 

# [Environment Requirements](#contents)

Framework

- [MindSpore](https://gitee.com/mindspore/mindspore)

For more information, please check the resources below：

- [MindSpore Tutorials](https://www.mindspore.cn/tutorials/en/master/index.html)
- [MindSpore Python API](https://www.mindspore.cn/docs/api/en/master/index.html)

# [Quick Start](#contents)

After installing MindSpore via the official website, you can start training and evaluation as follows:

```bash
python demo_ms.py
```
