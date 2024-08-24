When Deep Learning meets Van Gough

# Neural Style Transfer using VGG-19

This repository implements the paper **"A Neural Algorithm of Artistic Style"** by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge.

<p align="center">
  <img src="images/content_image.jpg" alt="Content Image" width="45%"/>
  <img src="images/style_image.jpg" alt="Style Image" width="45%"/>
</p>

<p align="center">
  <img src="images/output_image.jpg" alt="Output Image" width="90%"/>
</p>

<p align="center">
  <b>Figure:</b> (Left) Content Image, (Right) Style Image, (Below) Output Image
</p>

## Overview

This works by using a deep neural network to take two images—a content image and a feature image—and blend them together so that the output image looks like the content image but "painted" in the style of the style image. VGG-19, a 19-layer CNN trained on the ImageNet dataset, is utilized here. Since this implementation doesn't require training or fully connected layers, the trainable parameters are set to `False`.

To obtain a multi-level representation of the style image, feature correlations are calculated between different feature maps. The content loss between the content image and the output image is computed using the mean squared error (MSE) of localized shapes. However, for the style image, calculating loss through MSE is less effective, as the goal is to mimic the color and texture in the output image. Therefore, the Gram matrix is used to capture a more general representation of the style image.

```bash

def gm(tensor):
  '''Tensor multiplication with its own transpose, to spread and redistribute the original information across itself to remove localized data points.'''

  channels = int(tensor.shape[-1])
  a = tf.reshape(tensor, [-1, channels])
  gram = tf.matmul(a, a, transpose_a = True)
  return gram

``` 
During the training, gradient descent is employed to reduce the content and feature loss to create an output image.  


### Paper Reference
- **Title**: A Neural Algorithm of Artistic Style
- **Authors**: Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge
- **Link**: [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)

