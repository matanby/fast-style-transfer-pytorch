from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from fire import Fire
from torch import Tensor
# noinspection PyPep8Naming
from torch.nn import functional as F
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder
# noinspection PyPep8Naming
from torchvision.transforms import functional as T
from tqdm import tqdm

import image_utils
from config import TrainerConfig
from model import ImageTransformerModel
from vgg import Vgg19


class Trainer:
    def __init__(self, config: TrainerConfig, use_gpu: bool = True):
        self._config = config
        gpu_available = torch.cuda.is_available()
        self._device = 'cuda' if use_gpu and gpu_available else 'cpu'
        self._vgg = Vgg19(use_avg_pooling=True).to(self._device)
        self._create_model()
        self._opt = torch.optim.Adam(self._model.parameters(), config.learning_rate)
        self._create_data_loader()
        self._calc_style_gram_matrices()
        self._create_logdir()
        self._tensorboard = SummaryWriter(self._logdir)

    def train(self) -> None:
        c = self._config
        print(c)
        step = 0

        for epoch in range(c.epochs):
            prog_bar = tqdm(self._train_data_loader)
            for i, batch in enumerate(prog_bar):
                batch = batch[0].to(self._device)
                loss = self._step(batch)
                prog_bar.set_description(f'Train loss: {loss:.2f}')
                self._tensorboard.add_scalar('train/loss', loss, step)
                if i % c.visualization_interval == 0:
                    self._visualize_images(batch, step, 'train')
                if i != 0 and i % c.snapshot_interval == 0:
                    self._save_snapshot(step)
                step += 1

            prog_bar = tqdm(self._validation_data_loader)
            losses = []
            for i, batch in enumerate(prog_bar):
                batch = batch[0].to(self._device)
                with torch.no_grad():
                    loss = self._calc_loss(batch)
                losses.append(loss)
                prog_bar.set_description(f'Validation loss: {loss:.2f}')
            # noinspection PyUnresolvedReferences
            mean_loss = np.mean(losses)
            self._tensorboard.add_scalar('validation/loss', mean_loss, epoch)
            # noinspection PyUnboundLocalVariable
            self._visualize_images(batch, epoch, 'validation')
            self._save_snapshot(step)

    def _create_model(self) -> None:
        c = self._config
        self._model = ImageTransformerModel().train().to(self._device)
        if c.weights_snapshot_path:
            weights = torch.load(c.weights_snapshot_path)
            self._model.load_state_dict(weights)

    def _create_data_loader(self) -> None:
        c = self._config

        self._transform = transforms.Compose([
            transforms.Resize(c.input_images_dim, interpolation=Image.ANTIALIAS),
            transforms.RandomCrop(c.input_images_dim),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        train_set_path = Path(c.dataset_path) / 'train'
        dataset = ImageFolder(train_set_path, self._transform)
        self._train_data_loader = DataLoader(
            dataset=dataset,
            batch_size=c.batch_size,
            shuffle=True,
            num_workers=c.num_data_loader_workers,
            pin_memory=True,
            drop_last=True,
        )

        validation_set_path = Path(c.dataset_path) / 'validation'
        dataset = ImageFolder(validation_set_path, self._transform)
        self._validation_data_loader = DataLoader(
            dataset=dataset,
            batch_size=c.batch_size,
            num_workers=c.num_data_loader_workers,
            pin_memory=True,
            drop_last=True,
        )

    def _calc_style_gram_matrices(self) -> None:
        c = self._config
        style = image_utils.load(c.style_image_path)
        style_pil = image_utils.to_pil(style)
        # noinspection PyTypeChecker
        style_resized = T.resize(style_pil, c.input_images_dim)
        style_t = T.to_tensor(style_resized)
        style_t = style_t.to(self._device)

        self._style_gram_matrices = []
        with torch.no_grad():
            style_features = self._vgg(style_t)
            for features in style_features:
                gram_matrix = self._gram_matrix(features).repeat(c.batch_size, 1, 1)
                self._style_gram_matrices.append(gram_matrix)

    def _create_logdir(self) -> None:
        root_logdir = Path(self._config.root_logdir)
        style_image_name = Path(self._config.style_image_path)
        self._logdir = root_logdir / style_image_name.stem
        self._logdir.mkdir(parents=True, exist_ok=True)

    def _step(self, batch: Tensor) -> float:
        loss = self._calc_loss(batch)
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        loss_f = loss.item()
        return loss_f

    def _calc_loss(self, batch:  Tensor) -> Tensor:
        c = self._config
        with torch.no_grad():
            features_input = self._vgg(batch)
        transformed = self._model(batch)
        features_transformed = self._vgg(transformed)
        content_loss = self._content_loss(features_transformed, features_input)
        style_loss = self._style_loss(features_transformed)
        tv_loss = self._tv_loss(transformed)
        loss = content_loss * c.lambda_content + style_loss * c.lambda_style + c.lambda_tv * tv_loss
        return loss

    def _content_loss(self, features_input: List[Tensor], features_target: List[Tensor]) -> Tensor:
        total = torch.zeros(1, dtype=torch.float32, device=self._device)
        weights = self._config.content_block_weights

        num_features = len(features_input)
        for i in range(num_features):
            if weights[i] > 0:
                block_loss = F.mse_loss(features_input[i], features_target[i])
                block_loss = block_loss
                total = total + block_loss * weights[i]

        return total

    def _style_loss(self, features_input: List[Tensor]) -> Tensor:
        total = torch.zeros(1, dtype=torch.float32, device=self._device)
        weights = self._config.style_block_weights

        num_features = len(features_input)
        for i in range(num_features):
            if weights[i] > 0:
                gram_input = self._gram_matrix(features_input[i])
                gram_target = self._style_gram_matrices[i]
                block_loss = F.mse_loss(gram_input, gram_target)
                total = total + block_loss * weights[i]

        return total

    @staticmethod
    def _tv_loss(image: Tensor) -> Tensor:
        tv_loss = (image[:, :, :, :-1] - image[:, :, :, 1:]).abs().mean() + \
                  (image[:, :, :-1, :] - image[:, :, 1:, :]).abs().mean()

        return tv_loss

    @staticmethod
    def _gram_matrix(features: Tensor) -> Tensor:
        n, c, h, w = features.shape
        x = features.view(n, c, h * w)
        y = features.view(n, c, h * w).permute(0, 2, 1)
        gram = torch.bmm(x, y)
        gram = gram / (h * w)
        return gram

    def _visualize_images(self, batch: Tensor, step: int, tag: str) -> None:
        with torch.no_grad():
            transformed = self._model(batch)
        self._tensorboard.add_images(f'{tag}/inputs', batch, step)
        self._tensorboard.add_images(f'{tag}/transformed', transformed, step)

    def _save_snapshot(self, step: int) -> None:
        output_path = self._logdir / f'step_{step}.pt'
        torch.save(self._model.state_dict(), output_path)


def train(**kwargs):
    config = TrainerConfig(**kwargs)
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    Fire(train)
