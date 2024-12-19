from collections.abc import Iterable
from typing import Any, Tuple


class ConfigSection:
    """
    Stores a config section's variables.

    The config section has no formatting and type checking of config values. This is the job of the config itself. The
    config section just accepts whatever it is given.

    A config section will keep track of how many times a config parameter is gathered. So, if a config parameter is
    never used, this can be detected by the function `list_redundant_params`.

    Attributes:
        name (str): the name of the config section.
    """

    name: str

    _attributes: dict[str, Any]
    _retrieval_counts: dict[str, int]

    def __init__(self, name: str, values: dict[str, Any]) -> None:
        """
        Initialise a config section.

        Args:
            name (str): the name of the config section.
            values (dict[str, any]): the name of each parameter as a key, the value as its value in the config section.
        """
        assert type(name) is str
        assert type(values) is dict
        assert all([type(param_name) is str for param_name in values.keys()])
        assert len(values) > 0, "Config must contain at least one parameter"

        self.name = name
        self._attributes = {}
        self._retrieval_counts = {name: 0 for name in values.keys()}
        for name, val in values.items():
            self._attributes[name] = val

    def __getitem__(self, param_name: str) -> Any:
        """
        Allows a config parameter to be gathered by config_section["param_name"].
        """
        if type(param_name) is not str:
            raise TypeError(f"Config section parameters must be accessed individually by a str, got {type(param_name)}")
        if param_name not in self._attributes:
            raise ValueError(f"Could not find parameter {param_name} in {self.name} config section")

        self._retrieval_counts[param_name] += 1

        return self._attributes[param_name]

    def __setitem__(self, key: str, value: Any, /) -> None:
        """
        Allows a config parameter to be edited by config_section["param_name"] = value.
        """
        assert type(key) is str
        assert key in self._attributes.keys()

        self._retrieval_counts[key] = 0
        self._attributes[key] = value

    def items(self) -> Iterable[Tuple[str, Any]]:
        """
        List all the config section parameter names and values.

        Returns:
            (iterable[tuple[str, any]]): items. All config section parameter names and values.
        """
        # TODO: Remove this legacy function so the retrieval count of each config parameter can be correctly counted.
        for name in self._retrieval_counts.keys():
            self._retrieval_counts[name] += 1
        return self._attributes.items()

    def get_parameter_names(self) -> list[str]:
        """
        Get every parameter name in the config section.

        Returns a list containing every parameter name.
        """
        return list(self._attributes.keys())

    def list_redundant_params(self) -> tuple[str, ...]:
        """
        List parameters that were never retrieved in the config section.

        Returns:
            (list[int]): redundant_names. A list containing the names of each parameter never retrieved.
        """
        redundant_parameters = []
        for name, count in self._retrieval_counts.items():
            if count == 0:
                redundant_parameters.append(name)
        return tuple(redundant_parameters)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the config section parameters into a dictionary. Used for saving placing the config section inside of a
        notebook page.
        """
        return self._attributes.copy()
