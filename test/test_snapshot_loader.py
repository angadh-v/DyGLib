import pytest
import torch
from utils.DataLoader import SnapshotLoader


@pytest.mark.parametrize("num_workers", [1, 2])
def test_snapshot_loader(num_workers):
    time = torch.tensor([0, 0, 1, 1, 1, 1, 2, 3, 4, 4])

    loader = SnapshotLoader(time, horizon=1, num_workers=num_workers)
    assert len(loader) == 5
    for i, t in enumerate(loader):
        assert t.tolist() == torch.where(time == i)[0].tolist()

    loader = SnapshotLoader(time, horizon=2, num_workers=num_workers)
    assert len(loader) == 3
    for i, t in enumerate(loader):
        assert (
            t.tolist() == torch.where(
                (time == i * 2) | (time == i * 2 + 1)
            )[0].tolist()
        )

    # Test time data starting from 100 and with gaps
    time = torch.tensor([100, 100, 101, 101, 101, 101, 106, 106, 107, 107])

    loader = SnapshotLoader(time, horizon=1, num_workers=num_workers)
    assert len(loader) == 8
    i = 0
    for t in loader:
        while (time == (i + 100)).sum() == 0:
            i += 1
        assert t.tolist() == torch.where(time == (i + 100))[0].tolist()
        i += 1

    loader = SnapshotLoader(time, horizon=2, num_workers=num_workers)
    assert len(loader) == 4
    i = 0
    for t in loader:
        while ((time == (i * 2 + 100)) | (time == (i * 2 + 101))).sum() == 0:
            i += 1
        assert (
            t.tolist() == torch.where(
                (time == (i * 2 + 100)) | (time == (i * 2 + 101))
            )[0].tolist()
        )
        i += 1
