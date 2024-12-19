import shutil
import subprocess
import tempfile
import time
from os import path

import pytest

# Each test file has its own timeout. This way the testing is not hanging around for tests that never finish without
# user input. The bool flag is set to true if the docs example modifies the notebook.
TEST_FILE_NAMES: tuple[tuple[str, int, bool]] = (
    ("compatibility_tracker_stage_names", 20, False),
    ("compatibility_tracker_start_from", 20, False),
    ("docstring_example", 5, False),
    ("estimate_runtime", 7, False),
    ("export_to_pciseq_0", 20, False),
    ("export_to_pciseq_1", 20, False),
    ("generate_gene_codes", 20, False),
    ("nb_delete_page_0", 7, True),
    ("nb_delete_page_1", 7, True),
    ("retrieve_notebook_config", 20, False),
    ("run_pipeline_0", 10, False),
)
NB_REPLACE = '"/path/to/notebook"'
CONFIG_REPLACE = '"/path/to/config.ini"'
REPLACEMENTS = {
    "method": '"prob"',
    "n_rounds": "7",
    "n_channels": "9",
    "n_gene_codes": "11",
    "score_thresh": "0.05",
    "intensity_thresh": "0.05",
    "page_name": "basic_info",
}


def run_python_script(script_path, timeout):
    # Start the process and run the Python script
    process = subprocess.Popen(["python", script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the process to complete, but give it a timeout (in seconds).
    start_time = time.time()
    while True:
        if process.poll() is not None:
            break
        if time.time() - start_time > timeout:
            print("Timeout exceeded. Killing the process.")
            process.kill()
            return
        time.sleep(0.2)

    # Get the output and errors from the process
    stdout, stderr = process.communicate()

    if process.returncode != 0 and "EOFError" not in stderr.decode():
        raise AssertionError(f"Error occurred: {stderr.decode()}")
    else:
        print(f"Script output:\n{stdout.decode()}")


@pytest.mark.notebook
def test_all_docs() -> None:
    # Snippets of code used in the docs are integration tested to ensure that they do not crash when run. All code is
    # run with a timeout so code that hangs waiting for an input is not stuck forever.
    docs_dir = path.dirname(path.dirname(__file__))

    config_path = path.join(path.dirname(docs_dir), "robominnie", "test", ".integration_dir", "robominnie.ini")
    nb_path = path.join(path.dirname(config_path), "output_coppafish", "notebook")
    tmp_dir = tempfile.TemporaryDirectory(prefix="coppafish")

    with tempfile.TemporaryDirectory(suffix="coppafish_nb") as temp_dir:
        nb_path_copy = path.join(temp_dir, "notebook_copy")

        last_test_modified_nb = False
        for file_name, timeout, modifies_nb in TEST_FILE_NAMES:
            if last_test_modified_nb:
                # Make a copy of the notebook before trying our scripts on it, this way code that modifies the notebook
                # won't modify the original.
                if path.isdir(nb_path_copy):
                    shutil.rmtree(nb_path_copy)
                shutil.copytree(nb_path, nb_path_copy)

            script_file_path = path.join(docs_dir, file_name + ".py")
            if path.isdir(script_file_path):
                continue
            if not script_file_path.endswith(".py"):
                continue

            with open(script_file_path, "r") as file:
                code = "".join(file.readlines())

            # Replace any temp notebook paths with a real notebook path.
            code = code.replace(NB_REPLACE, f'r"{nb_path}"')
            code = code.replace(CONFIG_REPLACE, f'r"{config_path}"')
            for replace_content, replace_to in REPLACEMENTS.items():
                code = code.replace(replace_content, replace_to)

            new_script_path = path.join(tmp_dir.name, "python_temp_run.py")
            with open(new_script_path, mode="w") as new_file:
                new_file.write(code)

            run_python_script(new_script_path, timeout)
            last_test_modified_nb = modifies_nb

    tmp_dir.cleanup()
