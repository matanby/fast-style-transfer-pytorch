from dataclasses import dataclass
from typing import Tuple


@dataclass
class TrainerConfig:
    # the path to the folder containing training and validation sets.
    # this folder should be structured as follows:
    # - train
    #   - images
    #     - image_1.jpg
    #     - ...
    # - validation
    #   - images
    #     - image_1.jpg
    #     - ...
    dataset_path: str

    # the path to the target style image.
    style_image_path: str

    # the root directory in which model snapshots
    # and TensorBoard logs will be saved.
    root_logdir: str = 'models'

    # a path to a snapshot of the model's weights.
    # to be used when resuming a previous training job.
    weights_snapshot_path: str = ''

    # the weight of the content term in the total loss.
    # empirically good range: 1 - 100
    lambda_content: float = 10

    # the weight of the style term in the total loss.
    # empirically good range: 10 - 100_000
    lambda_style: float = 100

    # the weight of the generated image's total variation
    # in the total loss. empirically good range: 0 - 1_000.
    lambda_tv: float = 10

    # the size of each step of the optimization process.
    learning_rate: float = 1e-3

    # number of training epochs to perform.
    epochs: int = 2

    # the weight of each convolutional block in the content loss.
    # These five numbers refer to the following five activations of
    # the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
    content_block_weights: Tuple[float] = (0.0, 1.0, 0.0, 0.0, 0.0)

    # the weight of each convolutional block in the style loss.
    # These five numbers refer to the following five activations of
    # the VGG19 model: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1.
    style_block_weights: Tuple[float] = (1/5, 1/5, 1/5, 1/5, 1/5)

    # the dimension of the model's input images.
    input_images_dim: int = 256

    # the interval (number of training iterations) after which intermediate
    # results of the stylized images will be visualized in TensorBoard.
    visualization_interval: int = 50

    # the interval (number of training iterations) after which an
    # intermediate snapshot of the model will be saved to the disk.
    snapshot_interval: int = 1000

    # the mini batch size to use for each training iteration.
    batch_size: int = 4

    # the number of workers to use for loading images
    # from the dataset in the background
    num_data_loader_workers: int = 5

    def update(self, **kwargs) -> 'TrainerConfig':
        for key, value in kwargs.items():
            if key in self.__dict__:
                setattr(self, key, value)
            else:
                raise KeyError(f'Unknown configuration value: "{key}"')
        return self
