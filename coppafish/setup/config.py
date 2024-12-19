import configparser
import importlib.resources as importlib_resources
import re
from collections.abc import Callable
from os import path
from typing import Any, Dict, Tuple

from .. import log
from .config_section import ConfigSection

FORMATTED_PARAM_TYPE = (
    int | float | str | bool | None | Tuple[int, ...] | Tuple[float, ...] | Tuple[str, ...] | Tuple[bool, ...]
)


class Config:
    """
    Reads and validates config files.

    Every section is separated into ConfigSection classes with parameters that are accessed like a dictionary. For
    example, you can gather the variable "use_tiles" from basic_info by

    ```py
    use_tiles = config["basic_info"]["use_tiles"]
    ```

    (This allows for legacy code support as the config used to be a plain python dictionary object).
    """

    @staticmethod
    def get_default_for(section_name: str, parameter_name: str) -> FORMATTED_PARAM_TYPE:
        """
        Get the default value for a particular parameter. The parameter is pre-checked, formatted, and post-checked.

        Args:
            section_name (str): section name.
            parameter_name (str): parameter name.

        Returns:
            (any): parameter_default. The default value of the parameter. Can be None.
        """
        assert type(section_name) is str
        assert type(parameter_name) is str

        config = Config()
        parser = config.create_parser()
        config._parse_config(parser, config._get_default_file_path())
        if section_name not in parser.keys():
            raise ValueError(f"No config section called {section_name}")
        if parameter_name not in parser[section_name].keys():
            raise ValueError(f"No parameter called {parameter_name}")

        value = parser[section_name][parameter_name]
        pre_check_str = config._options[section_name][parameter_name][0]
        if not config.pre_check_param(parameter_name, section_name, value, pre_check_str):
            raise ValueError(f"Default parameter value {value} failed pre config check {pre_check_str}")
        formatted_value = config.format_param(parameter_name, section_name, value, pre_check_str)
        post_check_str = config._options[section_name][parameter_name][1]
        if not config.post_check_param(parameter_name, section_name, formatted_value, post_check_str):
            raise ValueError(f"Default parameter value {formatted_value} failed post config check {post_check_str}")

        return formatted_value

    _sections: list[ConfigSection]

    def get_sections(self) -> tuple[ConfigSection]:
        return tuple(self._sections)

    sections: Tuple[ConfigSection, ...] = property(get_sections)

    _current_section_index: int

    # Parameter pre checkers cannot be combined except for the "maybe" and "tuple" keyword, separated by an underscore.
    # The check must be true for the parameter to be valid. These checkers are applied BEFORE the parameter is formatted
    # into its correct type, so the parameter is still a string. "maybe" must be at the start, "tuple" must come after.
    _param_pre_checks: dict[str, Callable[[str], bool]] = {
        "maybe": lambda x: x == "",
        "int": lambda x: re.fullmatch(r"-?[0-9|_]+", x) is not None,
        "number": lambda x: re.fullmatch(r"-?[0-9]+(\.[0-9]+)?$", x) is not None,
        "bool": lambda x: re.fullmatch("true|false", x, re.IGNORECASE) is not None,
        "str": lambda x: len(x) > 0,
        "file": lambda x: len(x) > 0,
        "dir": lambda x: len(x) > 0,
    }

    # Parameter formatters are callables that convert the given parameter config string value into said format.
    _param_formatters: dict[str, Callable[[str], None | int | float | bool | str | tuple[Any, ...]]] = {
        "maybe": lambda x: None if x == "" else x,
        "tuple": lambda x: tuple([s.strip() for s in x.split(",") if s.strip() != ""]),
        "int": lambda x: int(x),
        "number": lambda x: float(x),
        "bool": lambda x: "true" in x.lower(),
        "str": lambda x: x,
        "file": lambda x: x,
        "dir": lambda x: x,
    }

    # Parameter post checkers can be combined by an underscore between them, in which case all of those checkers must
    # be true for the parameter to be valid. These checkers are applied AFTER the parameter is formatted into its
    # correct type. The post checks are not done if the parameter can be "maybe" and is None.
    _param_post_checks: dict[str, tuple[Callable[[Any], bool]], str] = {
        "positive": (lambda x: x > 0, "positive"),
        "negative": (lambda x: x < 0, "negative"),
        "not-positive": (lambda x: x <= 0, "<= 0"),
        "not-negative": (lambda x: x >= 0, ">= 0"),
        "lt1": (lambda x: x < 1, "< 1"),
        "lteq1": (lambda x: x <= 1, "<= 1"),
        # String checks.
        "str-not-empty": (lambda x: bool(x.strip()), "a non empty string"),
        "file-exists": (lambda x: path.isfile(x), "an existing file"),
        "dir-exists": (lambda x: path.isdir(x), "an existing directory"),
        "email-address": (
            lambda x: re.fullmatch(r"([A-Za-z0-9]+[.-_])*[A-Za-z0-9]+@[A-Za-z0-9-]+(\.[A-Z|a-z]{2,})+", x),
            "a valid email address",
        ),
        # Tuple checks.
        "tuple-not-empty": (lambda x: len(x) > 0, "a non empty tuple"),
        "tuple-len-multiple-3": (lambda x: len(x) % 3 == 0, "a tuple of length multiple of 3"),
    }
    # Tuple types in the config are separated by a comma. There must be no bracket encapsulation.
    _tuple_separator = ","
    _format_separator = "_"
    _checker_separator = "_"
    _options_type = Dict[str, Dict[str, Tuple[str, str]]]

    # Each key is a configuration section.
    # Each value is a dict with each key being a parameter name, the value is a tuple containing two strings. The first
    # specifies the format for pre-checks and formatting, the second specifies the post checks, like "positive". This
    # can be left empty.

    # If you change config options, update the coppafish/setup/default.ini file too.
    _options: _options_type = {
        "basic_info": {
            "use_tiles": ("maybe_tuple_int", "tuple-not-empty"),
            "use_rounds": ("maybe_tuple_int", "tuple-not-empty"),
            "use_channels": ("maybe_tuple_int", "tuple-not-empty"),
            "use_z": ("maybe_tuple_int", "tuple-not-empty"),
            "use_dyes": ("maybe_tuple_int", "tuple-not-empty"),
            "use_anchor": ("bool", ""),
            "anchor_round": ("maybe_int", "not-negative"),
            "anchor_channel": ("maybe_int", "not-negative"),
            "dapi_channel": ("maybe_int", "not-negative"),
            "dye_names": ("tuple_str", "tuple-not-empty"),
            "is_3d": ("bool", ""),
            "bad_trc": ("maybe_tuple_int", "tuple-len-multiple-3"),
            "ignore_first_z_plane": ("bool", ""),
            "reverse_tile_positions_x": ("bool", ""),
            "reverse_tile_positions_y": ("bool", ""),
            # From here onwards these are not compulsory to enter and will be taken from the metadata
            # Only leaving them here to have backwards compatibility as Max thinks the user should influence these
            "channel_camera": ("maybe_tuple_int", ""),
            "channel_laser": ("maybe_tuple_int", ""),
        },
        "file_names": {
            "notebook_name": ("str", "str-not-empty"),
            "input_dir": ("str", "str-not-empty"),  # The raw data does not need to accessible once extract is complete.
            "output_dir": ("str", "dir-exists_str-not-empty"),
            "tile_dir": ("str", "dir-exists_str-not-empty"),
            "round": ("maybe_tuple_str", "tuple-not-empty"),
            "anchor": ("maybe_str", "str-not-empty"),
            "raw_extension": ("str", "str-not-empty"),
            "raw_metadata": ("maybe_str", "str-not-empty"),
            "code_book": ("str", "file-exists"),
            "psf": ("maybe_str", "file-exists"),
            "fluorescent_bead_path": ("maybe_str", ""),
            "initial_bleed_matrix": ("maybe_str", "file-exists"),
            "omp_mean_spot": ("maybe_str", "file-exists"),
        },
        "notifications": {
            "log_name": ("str", "str-not-empty"),
            "minimum_print_severity": ("int", "not-negative"),
            "allow_notifications": ("bool", ""),
            "notify_on_crash": ("bool", ""),
            "notify_on_completion": ("bool", ""),
            "sender_email": ("maybe_str", "str-not-empty_email-address"),
            "sender_email_password": ("maybe_str", "str-not-empty"),
            "email_me": ("maybe_str", "str-not-empty_email-address"),
        },
        "extract": {
            "num_rotations": ("int", "not-negative"),
            "z_plane_mean_warning": ("number", "not-negative"),
        },
        "filter": {
            "force_cpu": ("bool", ""),
            "wiener_constant": ("number", "not-negative"),
            "wiener_pad_shape": ("tuple_int", "tuple-not-empty"),
        },
        "find_spots": {
            "auto_thresh_multiplier": ("number", "not-negative"),
            "radius_xy": ("int", "not-negative"),
            "radius_z": ("int", "not-negative"),
            "max_spots_percent": ("number", "not-negative"),
            "n_spots_warn_fraction": ("number", "not-negative"),
            "n_spots_error_fraction": ("number", "not-negative"),
        },
        "stitch": {
            "expected_overlap": ("number", "not-negative"),
        },
        "register": {
            # this parameter is for channel registration
            "bead_radii": ("maybe_tuple_number", ""),
            # these parameters are for round registration
            "sample_factor_yx": ("int", "positive"),
            "chunks_yx": ("int", "positive"),
            "overlap_yx": ("number", "not-negative"),
            "window_radius": ("int", "positive"),
            "smooth_sigma": ("tuple_number", "tuple-not-empty"),
            "flow_cores": ("maybe_int", "positive"),
            "flow_clip": ("maybe_tuple_number", "tuple-not-empty"),
            # these parameters are for icp
            "neighb_dist_thresh_yx": ("number", "positive"),
            "neighb_dist_thresh_z": ("maybe_number", "positive"),
            "icp_min_spots": ("int", "not-negative"),
            "icp_max_iter": ("int", "not-negative"),
        },
        "call_spots": {
            "gene_prob_threshold": ("number", "not-negative"),
            "target_values": ("maybe_tuple_number", "tuple-not-empty"),
            "d_max": ("maybe_tuple_int", "tuple-not-empty"),
            "kappa": ("maybe_number", "not-negative"),
            "concentration_parameter_parallel": ("number", ""),
            "concentration_parameter_perpendicular": ("number", ""),
        },
        "omp": {
            "max_genes": ("int", "positive"),
            "minimum_intensity": ("number", "not-negative"),
            "alpha": ("number", "not-negative"),
            "beta": ("number", "positive"),
            "dot_product_threshold": ("number", "not-negative"),
            "subset_pixels": ("maybe_int", "positive"),
            "force_cpu": ("bool", ""),
            "radius_xy": ("int", "positive"),
            "radius_z": ("int", "positive"),
            "score_threshold": ("number", "not-negative"),
        },
        "thresholds": {
            "intensity": ("maybe_number", ""),
            "score_ref": ("number", ""),
            "score_omp": ("number", ""),
            "score_prob": ("number", ""),
            "score_omp_multiplier": ("number", ""),
        },
        "reg_to_anchor_info": {
            "full_anchor_y0": ("maybe_number", ""),
            "full_anchor_x0": ("maybe_number", ""),
            "partial_anchor_y0": ("maybe_number", ""),
            "partial_anchor_x0": ("maybe_number", ""),
            "side_length": ("maybe_number", ""),
        },
    }

    def get_options(self) -> _options_type:
        return self._options

    def set_options(self, value) -> None:
        self._options = value
        self._validate_options()

    options: _options_type = property(get_options, set_options)

    def __init__(self) -> None:
        self._validate_options()
        self._sections = []
        self._current_section_index = 0

    def __getitem__(self, section_name: str) -> ConfigSection:
        """
        Allows a config section to be gathered by config["section_name"].
        """
        if type(section_name) is not str:
            raise TypeError(f"Config sections must be accessed through a string, got type {type(section_name)}")
        if len(self._sections) == 0:
            raise self.SectionError("Load must be called first to parse the config file")

        for section in self._sections:
            if section.name == section_name:
                return section

        raise ValueError(f"No config section named {section_name}")

    def get_section_names(self) -> list[str]:
        return [section.name for section in self._sections]

    def load(self, file_path: str, default_file_path: str | None = None, post_check: bool = True) -> None:
        """
        Load the configuration file from the given file path. Any unset configuration values are set to their default
        values in the default file.

        Args:
            file_path (str): the file path to the custom configuration file set by the user.
            default_file_path (str or none, optional): the default config values for every parameter. If None, then the
                default config file path at coppafish/setup/default.ini is used.
            post_check (bool, optional): run post-checks on config variables after loading them. Default: true.
        """
        assert type(file_path) is str
        if default_file_path is None:
            default_file_path = self._get_default_file_path()
        assert type(default_file_path) is str
        for config_path in (file_path, default_file_path):
            if not path.isfile(config_path):
                raise FileNotFoundError(f"Could not find config file at {path}")

        parser = self.create_parser()

        # Load the default settings first.
        self._parse_config(parser, default_file_path)

        # Overwrite the default settings if the user has specified them.
        self._parse_config(parser, file_path)

        # Ensure every section in _options exists in the config files.
        # Also, ensure there are no additional sections in the config files.
        self._check_config_sections(parser)

        # Ensure every parameter exists in _options.
        self._check_params_exist(parser)

        # Run pre-checks on the config parameters.
        self._pre_check_params(parser)

        # Format all the config parameters.
        formatted_parser = self.format_params(parser)

        if post_check:
            # Run post-checks on the config parameters.
            self._post_check_params(formatted_parser)

        # Append all config sections/parameters.
        for section in self._options.keys():
            section_values = {}
            for param_name, param_value in formatted_parser[section].items():
                section_values[param_name] = param_value

            self._sections.append(ConfigSection(section, section_values))

    def _check_config_sections(self, parser: configparser.ConfigParser) -> None:
        for section in self._options.keys():
            if section not in parser.keys():
                raise self.SectionError(f"No config section {section} found in config _options")

        for section in parser.keys():
            if section == configparser.DEFAULTSECT:
                continue
            if section not in self._options.keys():
                raise self.SectionError(f"Unexpected config section {section} that is not in _options.")

    def _check_params_exist(self, parser: configparser.ConfigParser) -> None:
        param_msg = self._create_param_msg()
        # Ensure every expected config parameter exists in the config files.
        for section in self._options.keys():
            for param_name in self._options[section].keys():
                if param_name not in parser[section].keys():
                    raise self.MissingParamError(f"Expected {param_msg.format(param_name, section)}")

    def _pre_check_params(self, parser: configparser.ConfigParser) -> None:
        param_msg = self._create_param_msg()
        for section in self._options.keys():
            for param_name, param_value in parser[section].items():
                if param_name not in self._options[section].keys():
                    log.warn(f"Unknown config {param_msg.format(param_name, section)}, ignoring it")
                    parser[section].__delitem__(param_name)
                    continue
                pre_checker_str = self._options[section][param_name][0]
                if not self.pre_check_param(param_name, section, param_value, pre_checker_str):
                    raise self.ParamError(
                        f"Failed check on {param_msg.format(param_name, section)}, expected type "
                        + f"{self._convert_pre_checker_to_readable(pre_checker_str)} but got {param_value}"
                    )

    def format_params(self, parser: configparser.ConfigParser) -> dict[str, dict[str, FORMATTED_PARAM_TYPE]]:
        """
        Format all parameters. The parser has all parameters as strings. These are formatted into their correct types.
        These are str, bool, int, float, tuple, and None.

        Args:
            parser (ConfigParser): the config parser containing every section and their parameters.
        """
        all_section_values = {}

        for section in self._options.keys():
            section_values = {}
            for param_name, param_value in parser[section].items():
                pre_checker_str = self._options[section][param_name][0]
                value = self.format_param(param_name, section, param_value, pre_checker_str)
                section_values[param_name] = value
            all_section_values[section] = section_values

        return all_section_values

    def _post_check_params(self, formatted_parser: dict[str, dict[str, FORMATTED_PARAM_TYPE]]) -> None:
        param_msg = self._create_param_msg()
        for section in self._options.keys():
            for param_name, param_value in formatted_parser[section].items():
                post_checker_str = self._options[section][param_name][1]
                check_passed, msg = self.post_check_param(param_name, section, param_value, post_checker_str)
                if not check_passed:
                    raise self.ParamError(
                        f"Expected {param_msg.format(param_name, section)} to be {msg}, but got {param_value}"
                    )

    def pre_check_param(self, name: str, section: str, value: str, checker_str: str) -> bool:
        """
        Check if the given parameter value is valid based on its post-check string.

        Args:
            name (str): the name of the parameter.
            section (str): the name of the section the parameter is in.
            value (str): the value of the parameter, after formatting.
            checker_str (str): the keyword checks for the parameter, separated by underscores. For example, "positive",
                "positive_lt1" to be positive and less than 1.

        Returns:
            (bool): valid. Whether the parameter is valid or not.
        """
        assert type(checker_str) is str
        assert checker_str

        if not checker_str:
            return True

        if checker_str.startswith("maybe_"):
            checker_str = checker_str[6:]
            if value.strip() == "":
                return True

        check_values = (value,)

        if checker_str.startswith("tuple_"):
            checker_str = checker_str[6:]
            check_values = self._param_formatters["tuple"](value)

        for check_value in check_values:
            for check_name in checker_str.split(self._checker_separator):
                if check_name not in self._param_pre_checks.keys():
                    raise ValueError(f"Unknown check {check_name} given to parameter {name} in {section} in _options")

                if not self._param_pre_checks[check_name](check_value.strip()):
                    return False
        return True

    def format_param(self, name: str, section: str, value: str, format_str: str) -> FORMATTED_PARAM_TYPE:
        """
        Format the given parameter value based on the given formatting string set for said parameter. For example, if
        the parameter is "0.1" the format_str is "number" then it is converted to a float.

        Args:
            name (str): the parameter's name.
            section (str): the config section the parameter is located in.
            value (str): the string representation of the parameter.
            format_str (str): the specified formatting for the parameter. Some examples are "maybe_int", "int",
                "number", "maybe_tuple_int", "tuple_str". Tuples cannot be nested like notebook page variables.

        Returns:
            (any): formatted_value. The value formatted into the correct type.
        """
        assert type(name) is str
        assert type(section) is str
        assert type(value) is str
        assert type(format_str) is str

        # Deal with weirder formatting first, if they exist.
        if "maybe" in format_str and value == "":
            assert format_str.startswith("maybe_") and format_str.count("maybe_") == 1
            return None
        formatted_value: tuple[str, ...] = (value,)
        keep_tuple = False
        if "tuple_" in format_str:
            assert format_str.count("tuple_") == 1
            formatted_value: tuple[str, ...] = self._param_formatters["tuple"](value)
            keep_tuple = True

        format_str = format_str.replace("maybe_", "").replace("tuple_", "")
        for sub_format in format_str.split(self._format_separator):
            if not sub_format:
                continue
            if sub_format not in self._param_formatters.keys():
                raise ValueError(f"Unknown format {sub_format} specified for config parameter {name} in {section}")
            new_formatted_value = tuple()
            for sub_value in formatted_value:
                try:
                    new_formatted_value += (self._param_formatters[sub_format](sub_value),)
                except ValueError as e:
                    raise self.ParamError(
                        f"Failed to format config parameter {name} in {section} with formatting {sub_format}. It has "
                        + f"value {sub_value}. Error given: {e}"
                    ) from e
            formatted_value: tuple[Any, ...] = new_formatted_value

        if not keep_tuple:
            assert len(formatted_value) == 1
            formatted_value = formatted_value[0]
        return formatted_value

    def post_check_param(self, name: str, section: str, value: Any, checker_str: str) -> Tuple[bool, str | None]:
        """
        Check if the given parameter value is valid based on its post-check string.

        Args:
            name (str): the name of the parameter.
            section (str): the name of the section the parameter is in.
            value (any): the value of the parameter, after formatting.
            checker_str (str): the keyword checks for the parameter, separated by underscores. For example, "positive",
                "positive_lt1" to be positive and less than 1.

        Returns:
            Tuple containing two items:
                (bool): valid. Whether the parameter is valid or not.
                (str or none): message. A description of what condition the parameter failed to meet. None if no fail.
        """
        assert type(checker_str) is str

        if not checker_str:
            return True, None

        if value is None:
            return True, None

        check_values = (value,)

        if type(value) is tuple:
            check_values = value
            for tuple_check in [check for check in checker_str.split(self._checker_separator) if "tuple-" in check]:
                if not self._param_post_checks[tuple_check][0](value):
                    return False, self._param_post_checks[tuple_check][1]
        else:
            assert "tuple-" not in checker_str, "Tuple post checks can only apply to tuple formatted types in _options"

        for check_value in check_values:
            for check_name in checker_str.split(self._checker_separator):
                if "tuple-" in check_name:
                    continue
                if check_name not in self._param_post_checks.keys():
                    raise ValueError(f"Unknown check {check_name} given to parameter {name} in {section} in _options")
                if not self._param_post_checks[check_name][0](check_value):
                    return False, self._param_post_checks[check_name][1]
        return True, None

    def create_parser(self) -> configparser.ConfigParser:
        parser = configparser.ConfigParser()
        # Case sensitive parser.
        parser.optionxform = str

        return parser

    def _convert_pre_checker_to_readable(self, pre_check_str: str) -> str:
        """
        Convert a pre-check string into a more readable format for debugging and user convenience.

        Args:
            pre_check_str (str): the pre check string.

        Returns:
            (str): pre_check_readable. An easier to read version of the pre-check.
        """
        if not pre_check_str:
            return "nothing"
        pre_check_readable = ""
        for sub_check in pre_check_str.split(self._checker_separator):
            if not sub_check:
                continue
            if sub_check == "maybe":
                continue
            if sub_check == "tuple":
                pre_check_readable += "a series of "
            else:
                pre_check_readable += sub_check
        if pre_check_str.startswith("maybe_"):
            pre_check_readable += " or empty"

        return pre_check_readable

    def _parse_config(self, parser: configparser.ConfigParser, file_path: str) -> None:
        assert type(file_path) is str

        with open(file_path, "r") as f:
            parser.read_string(f.read())

    def _create_param_msg(self) -> str:
        return "parameter {} in section {}"

    def _get_default_file_path(self) -> str:
        return str(importlib_resources.files("coppafish.setup").joinpath("default.ini"))

    def _validate_options(self) -> None:
        param_msg = self._create_param_msg()
        for section in self._options.keys():
            for param_name, param_value in self._options[section].items():
                if type(param_name) is not str:
                    raise TypeError(f"Parameter name in _options must be str, got {type(param_name)}")
                if not param_name:
                    raise ValueError(f"Blank parameter name in section {section}")
                if type(param_value) is not tuple:
                    raise TypeError(
                        f"All _options[section].values() must be tuple, got {type(param_value)} for "
                        + f"{param_msg.format(param_name, section)}"
                    )
                if len(param_value) != 2:
                    raise ValueError(
                        f"All _options[section].values() must be tuple of len 2, got len {len(param_value)} for "
                        + f"{param_msg.format(param_name, section)}"
                    )
                if not all([type(option) is str for option in param_value]):
                    raise ValueError(f"Values inside of tuple for {param_msg.format(param_name, section)} must be str")

    class ParamError(Exception):
        pass

    class MissingParamError(Exception):
        pass

    class SectionError(Exception):
        pass
