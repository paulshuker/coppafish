from coppafish.compatibility import CompatibilityTracker
from coppafish.utils import system


def test_CompatibilityTracker() -> None:
    tracker = CompatibilityTracker()
    tracker.check("0.10.7", "1.0.0")
    tracker.print_stage_names()

    assert tracker.has_version(
        system.get_software_version()
    ), "Require the latest software version inside of the coppafish/compatibility/base.py"
