import os
import tempfile

import numpy as np

from coppafish.setup.config import Config
from coppafish.setup.config_section import ConfigSection


def test_Config() -> None:
    assert Config.get_default_for("file_names", "notebook_name") == "notebook"

    config = Config()
    config.options.clear()
    config.options["debug"] = {
        "1": ("int", ""),
        "2": ("number", ""),
        "3": ("number", ""),
        "4": ("bool", ""),
        "5": ("file", "file-exists"),
        "6": ("dir", "dir-exists"),
        "7": ("tuple_int", ""),
        "8": ("tuple_number", ""),
        "9": ("tuple_number", ""),
        "10": ("tuple_bool", ""),
        "11": ("tuple_file", "file-exists"),
        "12": ("tuple_dir", "dir-exists"),
        "13": ("maybe_int", ""),
        "14": ("maybe_number", ""),
        "15": ("maybe_number", ""),
        "16": ("maybe_bool", ""),
        "17": ("maybe_file", "file-exists"),
        "18": ("maybe_dir", "dir-exists"),
        "19": ("maybe_tuple_int", ""),
        "20": ("maybe_tuple_number", ""),
        "21": ("maybe_tuple_number", ""),
        "22": ("maybe_tuple_bool", ""),
        "23": ("maybe_tuple_file", "file-exists"),
        "24": ("maybe_tuple_dir", "dir-exists"),
    }
    dir = os.path.dirname(__file__)
    tmpdir = tempfile.TemporaryDirectory(dir=dir)
    tmpdir2 = tempfile.TemporaryDirectory(dir=dir)
    tmpfile = tempfile.NamedTemporaryFile(dir=dir)
    tmpfile2 = tempfile.NamedTemporaryFile(dir=dir)
    config_filepath = os.path.join(tmpdir.name, "test_config.ini")

    # Build a default config file with all parameters wrongly assigned.
    default_config_filepath = os.path.join(tmpdir.name, "default.ini")
    default_config_content = "[debug]\n"
    for i in range(1, 25):
        default_config_content += f"{i}  =   "
        default_config_content += "kfhwjghe4534, 242ref\n"

    rng = np.random.RandomState(0)

    with open(default_config_filepath, "w") as file:
        file.write(default_config_content)

    config_content = "\n[debug]\n"
    for i in range(1, 25):
        expected_type = config.options["debug"][str(i)][0]
        if "maybe" in expected_type and rng.rand() > 0.9:
            config_content += f"{i} =    \n"
        elif "tuple_int" in expected_type:
            config_content += f"{i} = 0, 2, -4 \n"
        elif "tuple_number" in expected_type:
            config_content += f"{i} = 0, 2.203, -4.5   \n"
        elif "tuple_bool" in expected_type:
            config_content += f"{i} = true, TrUE, FaLsE, false, FALSE \n"
        elif "tuple_file" in expected_type:
            config_content += f"{i} =  {tmpfile.name}, {tmpfile2.name}, {tmpfile.name} \n"
        elif "tuple_dir" in expected_type:
            config_content += f"{i} = {tmpdir2.name}, {tmpdir.name}  \n"
        elif "int" in expected_type:
            config_content += f"{i} = {'' if rng.rand()>0.5 else '-'}35533  \n"
        elif "number" in expected_type:
            config_content += f"{i} = {'' if rng.rand()>0.5 else '-'}35533.32552  \n"
        elif "bool" in expected_type:
            config_content += f"{i} = TrUe\n"
        elif "file" in expected_type:
            config_content += f"{i} = {tmpfile.name}\n"
        elif "dir" in expected_type:
            config_content += f"{i} = {tmpdir2.name}\n"
        else:
            raise AssertionError("I forgot something")

    # Set only the ith parameter incorrectly and assert an error is raised when loading the config.
    for i in range(1, 25):
        wrong_param_name = str(i)
        ind_min = config_content.index(wrong_param_name + " =")
        ind_max = config_content.index("\n", ind_min)
        config_content_wrong = config_content[:ind_min]
        config_content_wrong += wrong_param_name + "   ="
        config_content_wrong += "sfefeg353sfe".replace(str(i), "")
        config_content_wrong += config_content[ind_max:]
        with open(config_filepath, "w") as config_file:
            config_file.write(config_content_wrong)
        # Expect as error here as the default config does not have the debug section.
        try:
            config.load(config_filepath, post_check=False)
            raise AssertionError("Expected SectionError")
        except config.SectionError:
            pass

        try:
            config.load(config_filepath)
            raise AssertionError("Expected SectionError")
        except config.SectionError:
            pass

        try:
            config.load(config_filepath, "/wrong/path")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass

        try:
            config.load(config_filepath, default_config_filepath)
            raise AssertionError(f"Expected ParamError with incorrect parameter {i}")
        except config.ParamError as e:
            assert f" {i} " in str(e)
            assert "debug" in str(e)

    tmpdir.cleanup()
    tmpdir2.cleanup()
    tmpfile.close()
    tmpfile2.close()
    del config, config_content, config_content_wrong, config_filepath, default_config_filepath
    del tmpdir, tmpdir2, tmpfile, tmpfile2

    tmpfile = tempfile.NamedTemporaryFile(dir=dir)
    tmpdir = tempfile.TemporaryDirectory(dir=dir)

    # Create a correct config file and ensure the formatted values are all correct.
    config = Config()
    config.options.clear()
    config.options["debug"] = {
        "1": ("int", ""),
        "2": ("number", ""),
        "a": ("str", ""),
        "4": ("bool", ""),
        "5": ("file", "file-exists"),
        "b": ("dir", "dir-exists"),
        "7": ("tuple_int", ""),
        "8": ("tuple_number", ""),
        "9": ("tuple_str", ""),
        "10": ("tuple_bool", ""),
        "sfegvdg": ("tuple_file", "file-exists"),
        "12": ("tuple_dir", "dir-exists"),
        "13": ("maybe_int", ""),
        "14": ("maybe_number", ""),
        "15": ("maybe_str", ""),
        "16": ("maybe_bool", ""),
        "17": ("maybe_file", "file-exists"),
        "18": ("maybe_dir", "dir-exists"),
        "19": ("maybe_tuple_int", ""),
        "20": ("maybe_tuple_number", ""),
        "21": ("maybe_tuple_str", ""),
        "22": ("maybe_tuple_bool", ""),
        "23": ("maybe_tuple_file", "file-exists"),
        "24": ("maybe_tuple_dir", "dir-exists"),
    }

    expected_values = []
    config_content = "[debug]\n"
    default_config_content = "[debug]\n"
    for param_name, checks in config.options["debug"].items():
        rnd_spaces = " " * rng.randint(5)
        rnd_spaces_2 = " " * rng.randint(5)
        default_config_content += f"{param_name}{rnd_spaces}={rnd_spaces_2}1\n "
        config_content += "\n "
        value_count = 1
        expecting_tuple = False
        if "tuple" in checks[0]:
            value_count = rng.randint(4) + 1
            expecting_tuple = True
        expected_value = tuple()
        config_content += f"{param_name} = "
        last_i = value_count - 1
        for i in range(value_count):
            if "int" in checks[0]:
                expected_value += (rng.randint(-11, 11),)
                config_content += f"{expected_value[i]}"
            elif "number" in checks[0]:
                expected_value += (rng.rand() * 10,)
                config_content += f"{expected_value[i]}"
            elif "str" in checks[0]:
                expected_value += ("StrIng with Spaces",)
                config_content += f"{expected_value[i]}"
            elif "bool" in checks[0]:
                expected_value += (True if rng.rand() > 0.5 else False,)
                config_content += f"{'TrUe' if expected_value[i] else 'faLSE'}"
            elif "file" in checks[0]:
                expected_value += (tmpfile.name,)
                config_content += f"{expected_value[i]}"
            elif "dir" in checks[0]:
                expected_value += (tmpdir.name,)
                config_content += f"{expected_value[i]}"
            else:
                raise AssertionError("I forgot something")
            if expecting_tuple and i != last_i:
                config_content += ", "
        config_content += " "
        if not expecting_tuple:
            expected_value = expected_value[0]
        expected_values.append(expected_value)

    config_filepath = os.path.join(tmpdir.name, "config.ini")
    default_config_filepath = os.path.join(tmpdir.name, "default.ini")

    with open(default_config_filepath, "w") as file:
        file.write(default_config_content)
    with open(config_filepath, "w") as config_file:
        config_file.write(config_content)

    config.load(config_filepath, default_config_filepath)

    for i, param_name in enumerate(config.options["debug"].keys()):
        assert config["debug"][param_name] == expected_values[i]

    del config_content, default_config_content

    # Check the post-checkers are working.
    config = Config()
    config.options.clear()
    config.options["debug"] = {
        "1": ("int", "positive"),
        "2": ("int", "negative"),
        "3": ("int", "not-positive"),
        "4": ("int", "not-negative"),
        "5": ("int", "lt1"),
        "6": ("int", "lteq1"),
        "7": ("number", "positive"),
        "8": ("number", "negative"),
        "9": ("number", "not-positive"),
        "10": ("number", "not-negative"),
        "11": ("number", "lt1"),
        "12": ("number", "lteq1"),
        "13": ("str", "str-not-empty"),
        "14": ("str", "file-exists"),
        "15": ("str", "dir-exists"),
        "16": ("str", "email-address"),
        "17": ("tuple_int", "tuple-not-empty"),
        "18": ("tuple_int", "tuple-len-multiple-3"),
        "19": ("tuple_number", "tuple-not-empty"),
        "20": ("tuple_number", "tuple-len-multiple-3"),
    }
    default_config_content = "\n".join([f"{name} = " for name in config.options["debug"].keys()])
    default_config_content = "[debug]\n" + default_config_content + "\n"
    with open(default_config_filepath, "w") as file:
        file.write(default_config_content)
    correct_config_content = f"""[debug]
    1 = 2
    2 = -4
    3 = 0
    4 = 1
    5 = 0
    6 = 1
    7 = 0.01
    8 = -12.4
    9 = -0.0
    10 = 0.1
    11 = 0.5567
    12 = 1.0000
    13 = a
    14 = {tmpfile.name}
    15 = {tmpdir.name}
    16 = example.email225@outlook.com
    17 = 0, 2
    18 = 1, 1, 2, 3, 4, 5
    19 = 0.5, 1, 2, 4.5
    20 = 0.5, 1, 2, 4.5, 5.6, 7
    """
    with open(config_filepath, "w") as config_file:
        config_file.write(correct_config_content)

    config.load(config_filepath, default_config_filepath)
    assert type(config.sections) is tuple
    assert len(config.sections) == 1
    assert type(config.sections[0]) is ConfigSection
    assert config.sections[0].name == "debug"

    incorrect_values = (
        "0",
        "0",
        "1",
        "-1",
        "1",
        "2",
        "-0.5",
        "1.2",
        "12.7",
        "-12.7",
        "1.1",
        "1.01",
        "",
        "/nonexistent/path/filename.txt",
        "/nonexistent/path/",
        "failed email.sdl@outlook.com",
        "",
        "0, 1",
        "",
        "0.5, 1.1",
    )
    for i, param_name in enumerate(config.options["debug"].keys()):
        substring = param_name + " = "
        index_min = correct_config_content.index(substring)
        index_max = correct_config_content.index("\n", index_min)
        broken_config_content = correct_config_content[: index_min + len(substring)]
        broken_config_content += incorrect_values[i]
        broken_config_content += correct_config_content[index_max:]
        with open(config_filepath, "w") as config_file:
            config_file.write(broken_config_content)

        try:
            config.load(config_filepath, default_config_filepath)
            raise AssertionError(f"Expected ParamError for parameter {param_name}")
        except Config.ParamError as e:
            assert " " + param_name + " " in str(e)

    tmpdir.cleanup()
    tmpfile.close()
