# Few-shot Image Classification

**Author: Pin-Ying Wu**

**Table of contents**
- Overview
- Code
- Result Analysis

## Overview
### Task
- Implement Prototypical Network and hallucinated data to classify image with limited training data, which learn a metric space to **compute distances to prototype representations of each class to do classification**.
- The objective of this classifier is to minimize the distance between the feature and its ground truth class prototype, and maximize the distance between the feature and the prototype of other classes.
- **Hallucination**: Add random noise sampled from Normal distribution on the input image to augment the training data.
- Reference : [Li et al., “Adversarial Feature Hallucination Networks for Few-Shot Learning”, CVPR 2020](https://arxiv.org/pdf/2003.13193.pdf)
<center>
<!-- ![] (asset/arch.png)-->
<!-- ![](https://i.imgur.com/OH8MMfi.png) -->
<img src=asset/arch.png width=80%><br> 
</center>

- For the improved model, we adopt a discriminator as the paper in the reference to help the model generate better hallucinated features. The hallucinator and the discriminator are trained with typical WGAN loss. The rest of the model is the baseline model.

### Dataset
- **Mini-ImageNet Dataset:**
The dataset consists of `84x84` RGB images in `80` classes.

## Code
### Prerequisites
```
pip install -r requirements.txt
```

### Data Preparation
```
bash ./get_dataset.sh
```

### Training


### Checkpoints
| baseline | improved |
|:---:|:---:|
| [baseline model](https://www.dropbox.com/s/o97xdgdfo3xnnm1/baseline.pkl?dl=1)  |  [improved model](https://www.dropbox.com/s/4i20vnruoc6aygx/improved.pkl?dl=1)  |

### Evaluation
```
python3 test_testcase.py --load <checkpoint> --has_hallucinator True --test_csv <path_to_test_images_csv> --test_data_dir <path_to_test_images_dir> --testcase_csv <path_to_test_cases> --output_csv <path_to_output_csv>
```

## Result Analysis
**Accuracy:**
|Model|   baseline  |   improved  |
|:-----------:|:-----------:|:-----------:|
|Accuracy| $47.73 \pm 0.88 \%$ |  $48.34 \pm 0.88 \%$  |

### Experiments
#### Experiments of Different Distance Functions (Under 5-way 1-shot)
- Reference : [Snell et al., “Prototypical Networks for Few-shot Learning”, NIPS 2017](https://arxiv.org/pdf/1703.05175.pdf)

|K value|   Euclidean distance  |   cosine similarity  |   parametric function (MLP)  |
|:-----------:|:-----------:|:-----------:|:-----------:|
|Accuracy| $45.53 \pm 0.83 \%$ |  $39.25 \pm 0.81 \%$  |  $35.09 \pm 0.70 \%$  |

- As described in the reference paper, the objective of this classifier is to minimize the distance between the feature and its ground truth class prototype, and maximize the distance between the feature and the prototype of other classes. Euclidean distance, cosine similarity and parametric function are three derivative distance functions considered. Euclidean distance computes the L2 distance between two features, while cosine similarity looks at the angle between two vectors. Different from Euclidean distance, the larger the value of cosine similarity represents the closer of the two features. The table above presents the accuracy of the prototypical network using 3 different distance functions. We can observe that using Euclidean distance function achieves the best accuracy among 3 different variations.

#### Experiments of Different K-shot Setting (5-way K-shot)

|K value|   K = 1  |   K = 5  |   K = 10  |
|:-----------:|:-----------:|:-----------:|:-----------:|
|Accuracy| $39.6 \pm 0.8 \%$ |  $60.8 \pm 0.7 \%$  |  $66.8 \pm 0.6 \%$  |

- The table above shows the classification accuracy under different shots (i.e. K). We can observe that the accuracy increases as more support samples are used to compute the class prototype (i.e. larger K). For each class, the prototype is the mean of the feature of support samples. More support samples means the prototype is more robust to outliers and more representative of the class, but not as noisy as using a single support sample as the class prototype. Imagine we choose an outlier as the support sample, which is far away from other samples from the same class, then the prototype of this class will be very wrong. Hence, we think this is the reason why the model performs better for larger Ks. From the table, we can also observe that the improvement of the accuracy from K=1 to K=5 is larger than from K=5 to K=10. We think it is because using single-shot (K=1) as the prototype is too noisy, and both K=5 and K=10 average the support features over the samples, so the improvement saturates.

#### Experiments of Different M-augmentation Setting (5-way 1-shot)

|M value|   M = 10  |   M = 50  |   M = 100  |
|:-----------:|:-----------:|:-----------:|:-----------:|
|Accuracy| $37.72 \pm 0.74 \%$ |  $39.41 \pm 0.70 \%$  |  $42.91 \pm 0.74 \%$  |
- In this problem, we use the hallucinator to create augmented samples as the support samples, and keep the query samples the same. From the table above, we can observe that the more data for augmentation (i.e. larger M), the better the model can learn. As discussed before, more support samples for computing the class prototypes makes the model more robust to outliers. As a result, the model performs better when M is larger.





#### Baseline Model
<center>
<!-- ![] (asset/tSNE-baseline.png)-->
<!-- ![](https://i.imgur.com/PDWJs9h.png) -->
<img src=asset/tSNE-baseline.png width=60%><br> 
</center>

- In the t-SNE plot, the features of each data sample are plotted with the color corresponding to its label. Real data are shown as crosses and hallucinated data are shown as triangles. For a classification task, we hope the sample from the same class are clustered together, and not mixed with the samples from other classes. We can observe that the data in the latent space is roughly divided into five categories. However, they are not separated very much. For example, the samples of the blue class and black class are almost overlapped. The hallucinated data and real data are also distributed very closed.

#### Improved Model

- In the t-SNE plot, the features of each data sample are plotted with the color corresponding to its label. Compared with the baseline model, the data in the same group here are clustered closer, and the data in different groups are further apart. Therefore, the classification performance is better than the baseline model.
<center>
<!-- ![] (asset/tSNE-improved.png)-->
<!-- ![](https://i.imgur.com/tBnYehD.png) -->
<img src=asset/tSNE-improved.png width=60%><br> 
</center>


- As discussed in the baseline model, the hallucinated data can help the model learn better for the classification task. In this question, I adopt the concept of GAN as the paper in the reference, using a discriminator to help the hallucinator generate better features, where the hallucinator plays the role of a generator. The discriminator aims at distinguishing real data and hallucinated data, while the hallucinator aims at generating real-like data that can cheat the discriminator. As a typical GAN, ideally, the discriminator should not be able to distinguish between the real data and the hallucinated data, but it is separated in the t-SNE plot above, which means that the generator did not learn very well. Hence, the accuracy does not improve very much. However, from this plot, we can observe that samples from the same class are clustered better than that in the baseline model. As a result, the classification performance still improves.