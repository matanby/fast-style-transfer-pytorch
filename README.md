# Fast Neural Style-Transfer PyTorch

This is a simple and minimalistic PyTorch implementation of the fast neural style transfer method introduced in
[Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155) 
by Johnson et. al (2016).

The original neural style transfer method by Gatys et. al
([A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576)) generates the stylized image by 
iteratively optimizing the target loss function, which combines the content and style terms.

Unlike this method, which is slow by its nature, the method by Johnson et. al presents a method for training
a convolutional neural-network that takes in a content image and generates a stylized version of it.
This makes the image generation process orders of magnitudes faster. However, the down-side is that the network
is trained on one specific style, and therefore is not generic. 

<div align="center">
 <img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/11.jpg" height="500px">
</div>

---

### Prerequisites:
* Python 3
* CUDA + CUDNN (for GPU acceleration)

### Installation:
1. Clone this repository:
```
git clone https://github.com/matanby/fast-style-transfer-pytorch
```

2. Install PIP requirements:
```shell script
python3 -m virtualenv .venv
source .venv/bin/activate 
pip install -r fast-style-transfer-pytorch/requirements.py
```

---

### Usage:
You can use one of the three pre-trained models that are bundled with this repository, 
or train your custom models on your own style images (see training instructions below).
#### From command-line:
```shell script
python run.py [PATH_TO_PRETRAINED_MODEL] [PATH_TO_CONTENT_IMAGE] [PATH_TO_STYLIZED_OUTPUT]
```

#### Programmatically:

Use the `Stylizer` class to create stylized images programmatically. For example:
```python
import image_utils
from stylizer import Stylizer

stylizer = Stylizer('models/style1.pt')
image = image_utils.load('images/content/1.jpg')
stylized = stylizer.stylize(image)
image_utils.save(stylized, f'images/stylized/style1/1.jpg')
```

---

### Training the model:

You can train a custom model on your own inputs style images.
To do so, you'll need a dataset of content images to train on. 
The authors of the paper used the [MS-COCO 2014 dataset](https://cocodataset.org/#download).

To initiate the training process, run the `train.py` script as follows:
```bash
python train.py --dataset_path [PATH_TO_DATASET] --style_image_path [PATH_TO_STYLE_IMAGE]
```
See below for more info on how the dataset folder should be structured. 

It is also possible to override the default configuration entries and hyper-parameters values,
 by providing additional CLI arguments, for example:
 
```shell script
python train.py \
  --dataset_path ms-coco \
  --style_image_path images/style/1.jpg \
  --batch_size 8 \
  --lambda_style 200
```

Complete list of configuration entries and hyper-parameters:    
* `dataset_path`: the path to the folder containing training and validation sets.
   this folder should be structured as follows:
   - train
     - images
       - image_1.jpg
       - ...
   - validation
     - images
       - image_1.jpg
       - ...
* `style_image_path`: the path to the target style image.
* `root_logdir`: the root directory in which model snapshots and 
   TensorBoard logs will be saved. default = 'models'.   
* `weights_snapshot_path`: a path to a snapshot of the model's weights.
   to be used when resuming a previous training job. default = ''.
* `lambda_content`: the weight of the content term in the total loss.
   empirically good range: 1 - 100. default = 10.
* `lambda_style`: the weight of the style term in the total loss.
  empirically good range: 10 - 100,000. default = 100.
* `lambda_tv`: the weight of the generated image's total variation 
   in the total loss. empirically good range: 0 - 1,000. default = 10.
* `learning_rate`: the size of each step of the optimization process. default = 1e-3.
* `epochs`: number of training epochs to perform. default = 2.
* `content_block_weights`: the weight of each convolutional block in the content loss.
   These five numbers refer to the following five activations of
   the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
   default = (0.0, 1.0, 0.0, 0.0, 0.0).
* `style_block_weights`: the weight of each convolutional block in the style loss.
   These five numbers refer to the following five activations of
   the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
   default = (1/5, 1/5, 1/5, 1/5, 1/5).
* `input_images_dim`: the dimension of the model's input images. default = 256.
* `visualization_interval`: the interval (number of training iterations) 
   after which intermediate results of the stylized images will be visualized 
   in TensorBoard. default = 50.
* `snapshot_interval`: the interval (number of training iterations) after which an
   intermediate snapshot of the model will be saved to the disk. default = 1000.
* `batch_size`: the mini batch size to use for each training iteration. default = 4.
* `num_data_loader_workers`: the number of workers to use for loading images
   from the dataset in the background. default = 5.
---

### Examples:
<div align="center">
<table>
<tr>


</tr>
<tr>
<td vlign="center"><center>Content Image / </br> Style Image</center></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/style/1.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/style/2.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/style/3.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/content/1.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/1.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style2/1.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style3/1.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/content/2.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/2.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style2/2.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style3/2.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/content/3.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/3.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style2/3.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style3/3.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/content/4.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/4.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style2/4.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style3/4.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/content/5.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/5.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style2/5.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style3/5.jpg" alt="content" width="200"/></td>
</tr>
<tr>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/content/6.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style1/6.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style2/6.jpg" alt="content" width="200"/></td>
<td vlign="center"><img src="https://github.com/matanby/fast-style-transfer-pytorch/raw/master/images/stylized/style3/6.jpg" alt="content" width="200"/></td>
</tr>
</table>
</div>
