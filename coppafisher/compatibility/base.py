from collections import OrderedDict

from .. import log
from ..setup.notebook import Notebook
from ..utils import system

# Each pipeline stage, this is slightly different to each coppafisher page.
STAGES: list[str] = [
    "initialisation",
    "extract",
    "filter",
    "find_spots",
    "register",
    "stitch",
    "ref_spots",
    "call_spots",
    "omp",
    "none",
]
# The page names associated with each stage. For multiple pages, they must be separated by a comma followed by a
# space.
PAGE_NAMES: tuple[str] = (
    "basic_info and file_names",
    "extract",
    "filter and filter_debug",
    "find_spots",
    "register and register_debug",
    "stitch",
    "ref_spots",
    "call_spots",
    "omp",
    "none",
)


class CompatibilityTracker:
    # For each coppafisher version, this is the earliest stage that requires re-running as a result of the changes is
    # given relative to the version before.
    # NOTE: This must be appended to for each future version release.
    _version_compatibility: OrderedDict[str, str] = OrderedDict(
        (
            ("0.10.6", "extract"),
            ("0.10.7", "filter"),
            ("0.10.8", "none"),
            ("1.0.0", "initialisation"),
            ("1.1.0", "none"),
            ("1.2.0", "none"),
        )
    )
    # For each stage, instructions are given on how to remove all data during and after said stage.
    _stage_instructions: list[tuple[str]] = [
        (
            "Clear the output directory. Delete the notebook",
            "Delete the 'extract' subdirectory inside of the 'tiles' directory",
        ),
        (
            "Clear the output directory. Delete the notebook",
            "Delete the 'extract' subdirectory inside of the 'tiles' directory",
        ),
        ("Clear the output directory. Delete the notebook",),
        (
            "Clear the output directory except the notebook",
            f"Remove notebook page {PAGE_NAMES[3]} and all later pages",
        ),
        (
            "Clear the output directory except the notebook",
            f"Remove notebook page {PAGE_NAMES[4]} and all later pages",
        ),
        (
            "Clear the output directory except the notebook",
            f"Remove notebook page {PAGE_NAMES[5]} and all later pages",
        ),
        (
            "Clear the output directory except the notebook",
            f"Remove notebook page {PAGE_NAMES[6]} and all later pages",
        ),
        (
            "Clear the output directory except the notebook",
            f"Remove notebook page {PAGE_NAMES[7]} and all later pages",
        ),
        (
            "Clear the output directory except the notebook",
            f"Remove notebook page {PAGE_NAMES[8]} and all later pages",
        ),
        ("Do nothing",),
    ]

    def __init__(self) -> None:
        assert len(PAGE_NAMES) == len(STAGES)
        assert len(STAGES) == len(self._stage_instructions)

    def check(self, from_version: str, to_version: str) -> None:
        """
        Check what output files can be kept when migrating between software versions.

        Args:
            - from_version (str): old software version.
            - to_version (str): software version migrating to.
        """
        assert type(from_version) is str
        assert type(to_version) is str
        assert from_version != to_version
        saved_versions_msg = f"valid versions are {', '.join(self._version_compatibility.keys())}"
        if not self.has_version(from_version):
            raise ValueError(f"Could not find version {from_version}, {saved_versions_msg}")
        if not self.has_version(to_version):
            raise ValueError(f"Could not find version {to_version}, {saved_versions_msg}")

        # Find the earliest stage changed from the versions after from_version up to to_version.
        earliest_stage, _ = self._get_earliest_stage_between(from_version, to_version)

        # Find and print the instructions to migrate from the earliest stage.
        log.info(f"Migrating from coppafisher {from_version} to {to_version}:")
        self.print_start_from(earliest_stage)
        log.info(
            "To delete notebook pages, see Usage -> Advanced Usage at "
            + "https://paulshuker.github.io/coppafisher/advanced_usage/#delete-notebook-page in the documentation"
        )

    def notebook_is_compatible(self, nb: Notebook) -> bool:
        """
        Check if the notebook contains incompatible data from older software versions. If so, a warning is printed and
        false is returned.

        Args:
            nb (Notebook): the notebook.

        Returns:
            (bool): valid. Whether all the notebook's data is compatible for this version of coppafisher.
        """
        assert type(nb) is Notebook

        current_version = system.get_software_version()
        for page_name, page_version in nb.get_all_versions().items():
            _, page_stage_index = self._get_stage_with_page_name(page_name)
            if page_version not in self._version_compatibility:
                raise ValueError(f"Notebook page {page_name} has unknown software version: {page_version}")
            # For a page and its version, if the earliest stage page that must be removed from the notebook for current
            # compatibility is equal to or earlier than this page, then the notebook is invalid.
            earliest_stage, earliest_stage_index = self._get_earliest_stage_between(page_version, current_version)
            if earliest_stage_index <= page_stage_index:
                # The notebook has backwards incompatibilities.
                log.warn(
                    f"The existing notebook contains backwards incompatibility on page {page_name} at version "
                    + f"{page_version} compared to current version {current_version}."
                )
                log.warn("The suggested course of action is:")
                self.print_start_from(earliest_stage)
                return False
        return self._check_notebook_for_downgrade(nb)

    def print_start_from(self, stage: str) -> None:
        """
        Print the instructions on how to prepare the coppafisher pipeline to start from the given stage.

        Args:
            stage (str): the stage to start again from.
        """
        [log.info(f"    - {instruction}.") for instruction in self._stage_instructions[STAGES.index(stage)]]

    def print_stage_names(self) -> None:
        """
        Print every stage that is part of coppafisher's pipeline in chronological order.
        """
        stage_names = []
        for stage_name in STAGES:
            if stage_name == "none":
                continue
            stage_names.append(stage_name)
        print(f"Coppafisher stages: {', '.join(stage_names)}")

    def has_version(self, version: str) -> bool:
        """
        Check if a version exists in the tracker's records.

        Args:
            version (str): the version to look for.

        Returns:
            (bool): version_exists. Whether the version exists.
        """
        return version in self._version_compatibility

    def _check_notebook_for_downgrade(self, nb: Notebook) -> bool:
        """
        Check the given notebook for a version drop when going through the stages, which should be impossible. We do
        not support users reverting back versions of the software and continuing a pipeline run as this has
        unpredictable consequences.

        Args:
            - nb (Notebook): the notebook to check. Does not have to contain all pages.

        Returns:
            (bool) valid: true if the software version did not downgrade.
        """
        current_version = system.get_software_version()
        current_version_index = list(self._version_compatibility.keys()).index(current_version)
        for page_name, page_version in nb.get_all_versions().items():
            page_version_index = list(self._version_compatibility.keys()).index(page_version)
            if page_version_index > current_version_index:
                log.warn(
                    f"Notebook contains page {page_name} run on version {page_version} which is higher than the "
                    + f"current coppafisher version ({current_version})."
                )
                log.warn("The suggested course of action is:")
                log.warn(f"    - Update coppafisher version to >= {page_version} before re-running.")
                return False
        return True

    def _get_earliest_stage_between(self, from_version_exclusive: str, to_version_inclusive: str) -> tuple[str, int]:
        index_start = list(self._version_compatibility.keys()).index(from_version_exclusive) + 1
        index_end = list(self._version_compatibility.keys()).index(to_version_inclusive) + 1
        earliest_stage = "none"
        earliest_stage_index = 999
        for index in range(index_start, index_end):
            stage = list(self._version_compatibility.values())[index]
            index = STAGES.index(stage)
            if index < earliest_stage_index:
                earliest_stage = stage
                earliest_stage_index = index
        return earliest_stage, earliest_stage_index

    def _get_stage_with_page_name(self, page_name: str) -> tuple[str, int]:
        for i, page_names in enumerate(PAGE_NAMES):
            if page_name in page_names.split(" and "):
                return STAGES[i], i
