from collections.abc import Callable
from typing import Optional


class Hotkey:
    def __init__(
        self,
        name: str,
        key_press: str,
        description: str,
        invoke: Optional[Callable],
        section: str,
        requires_selection: bool = True,  # If true, requires a selected spot in the viewer to work.
    ):
        self.name = name
        self.key_press = key_press
        self.description = description
        self.invoke = invoke
        self.section = section
        self.requires_selection = requires_selection

    def __str__(self):
        msg = "("
        if self.requires_selection:
            msg += "Select Spot, "
        msg += f"Press {self.key_press.lower().replace('-', ' + ')}) {self.name}"
        if self.description:
            msg += f": {self.description}"
        return msg
