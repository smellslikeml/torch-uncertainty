import pytest
import torch

from torch_uncertainty.transforms import SuperpixelMix


@pytest.fixture
def batch_input() -> tuple[torch.Tensor, torch.Tensor]:
    imgs = torch.rand(2, 3, 32, 32)
    targets = torch.randint(0, 5, (2, 32, 32))
    return imgs, targets


class TestSuperpixelMix:
    """Testing the SuperpixelMix transform."""

    def test_superpixelmix_shapes(self, batch_input) -> None:
        transform = SuperpixelMix(n_segments=10, compactness=10.0, prop=0.5)
        out_imgs, out_targets = transform(*batch_input)

        assert out_imgs.shape == batch_input[0].shape
        assert out_targets.shape == batch_input[1].shape

    def test_superpixelmix_integer_targets(self, batch_input) -> None:
        transform = SuperpixelMix(n_segments=10, compactness=10.0, prop=0.5)
        _, out_targets = transform(*batch_input)

        assert out_targets.dtype == batch_input[1].dtype
        assert not out_targets.dtype.is_floating_point

    def test_superpixelmix_targets_from_paired_maps(self, batch_input) -> None:
        imgs, targets = batch_input
        transform = SuperpixelMix(n_segments=10, compactness=10.0, prop=0.5)
        _, out_targets = transform(imgs, targets)

        next_targets = torch.roll(targets, shifts=-1, dims=0)
        for i in range(targets.shape[0]):
            kept = (out_targets[i] == targets[i]) | (out_targets[i] == next_targets[i])
            assert kept.all()

    def test_superpixelmix_changes_images(self, batch_input) -> None:
        imgs, targets = batch_input
        transform = SuperpixelMix(n_segments=10, compactness=10.0, prop=0.5)
        out_imgs, _ = transform(imgs, targets)

        assert not torch.allclose(out_imgs, imgs)

    def test_superpixelmix_invalid_prop(self) -> None:
        with pytest.raises(ValueError):
            _ = SuperpixelMix(prop=1.5)

    def test_superpixelmix_invalid_n_segments(self) -> None:
        with pytest.raises(ValueError):
            _ = SuperpixelMix(n_segments=0)

    def test_superpixelmix_invalid_compactness(self) -> None:
        with pytest.raises(ValueError):
            _ = SuperpixelMix(compactness=-1.0)
