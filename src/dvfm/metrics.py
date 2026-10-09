import torch


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> float:
    """Energy distance between two point clouds of shape (N, D) and (M, D).

    Close to 0 when x and y come from the same distribution; larger means more different.
    """
    xy = torch.cdist(x, y).mean()  # mean distance between the two clouds
    xx = torch.cdist(x, x).mean()  # mean distance within x
    yy = torch.cdist(y, y).mean()  # mean distance within y
    return (2 * xy - xx - yy).item()