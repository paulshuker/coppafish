import numpy as np
import torch

from .. import scores


def test_score_pixel_score_image() -> None:
    im_y, im_x, im_z = 4, 5, 6
    spot_shape = 1, 3, 5

    pixel_score_image = torch.zeros((2, im_y, im_x, im_z), dtype=torch.float32)
    pixel_score_image[1, 1, 3, 2] = 0.8
    mean_spot = torch.zeros(spot_shape, dtype=torch.float32)
    mean_spot[0, 1, 2] = 0.5
    mean_spot[0, 2, 2] = 0.9
    mean_spot[0, 1, 3] = 0.1

    spot_scores = scores.score_pixel_score_image(pixel_score_image, mean_spot)

    assert spot_scores.shape == pixel_score_image.shape
    assert torch.allclose(spot_scores[0], torch.zeros(1).float())
    assert torch.isclose(spot_scores[1, 1, 3, 2], 0.8 * 0.5 / mean_spot.sum())
    assert torch.isclose(spot_scores[1, 0, 0, 0], torch.asarray([0], dtype=torch.float32))


def test_boost_z_edge_spot_scores() -> None:
    im_y, im_x, im_z = 9, 10, 11
    rng = np.random.RandomState(0)
    spot_score_image = torch.from_numpy(rng.rand(2, im_y, im_x, im_z)).float()

    spot_shape = 3, 3, 5
    mean_spot = torch.ones(spot_shape).float()
    mean_spot[0, 0, 0] = 0.1
    mean_spot[0, 0, 1] = 0.3
    mean_spot[0, 2, 1] = 1.3
    mean_spot[2, 0, 2] = 0.2
    mean_spot[1, 0, 2] = 0.55
    mean_spot /= mean_spot.sum()

    spot_score_image_boosted = scores.boost_z_edge_spot_scores(spot_score_image, mean_spot)
    assert type(spot_score_image_boosted) is torch.Tensor
    assert spot_score_image_boosted.shape == spot_score_image.shape
    assert (spot_score_image_boosted >= spot_score_image).all()
    assert torch.isclose(spot_score_image_boosted, spot_score_image)[:, :, :, 2:-2].all()
    assert ((spot_score_image_boosted > spot_score_image)[:, :, :, :2]).all()
    assert ((spot_score_image_boosted > spot_score_image)[:, :, :, -2:]).all()
    assert torch.allclose(spot_score_image_boosted[..., 0], spot_score_image[..., 0] / (1 - mean_spot[:, :, :2].sum()))
    assert torch.allclose(spot_score_image_boosted[..., 1], spot_score_image[..., 1] / (1 - mean_spot[:, :, :1].sum()))
    assert torch.allclose(
        spot_score_image_boosted[..., -1], spot_score_image[..., -1] / (1 - mean_spot[:, :, :2].sum())
    )
    assert torch.allclose(
        spot_score_image_boosted[..., -2], spot_score_image[..., -2] / (1 - mean_spot[:, :, :1].sum())
    )
