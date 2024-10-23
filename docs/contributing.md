## The Algorithm

Coppafish is built on the basis that an algorithm that performs well does not need to be changed. Algorithms are only 
updated when there is evidence that it can perform better and that the current algorithm is performing worse.

Typically there is a protected staging branch, like `v1.0.0`, for a future release. This must be pull requested into 
and must pass continuous integration tests. The `main` branch remains the latest version release to be the default 
branch when users git clone the software.

## Run Tests

In your coppafish environment, install dev packages 

```terminal
pip install -r requirements-dev.txt
```

Run unit tests (~25s) 

```terminal
pytest -m "not integration and not manual and not notebook"
```

Run integration tests (~50s) 

```terminal
pytest -m "integration and not manual"
```

Run unit tests requiring a notebook (~3s) 

```terminal
pytest -m "notebook and not integration and not manual"
```

Check the percentage code coverage by appending `--cov=coppafish --cov-report term` to each command.

## Code Philosophy

We follow basic rules when coding. Anyone can code something that works, but coding it in a scaleable, maintainable way 
is another struggle altogether.

Here are some specific standards to follow:

* Knowledge written down twice is bad code. Don't Repeat Yourself (DRY)!
* If a bug is found, the bug must be automatically found if it is to occur again.
* Every time a function is modified or created, a new unit test must be created for the function. A pre-existing unit 
test can be drawn from to build a new unit test, but it should be clear in your mind that you are affectively building 
a new function.
* Minimise `#!python if`/`#!python else` branching as much as possible. Exit `#!python if`/`#!python else` nesting as 
soon as possible through the use of keywords like `#!python continue`, `#!python break` and `#!python return`, whenever 
feasible.
* Do not over-shorten a variable or function name.
* Variables and functions are not capitalised, classes are.
* In most cases, a line of code should do only one operation.
* Every docstring for a function must be complete so a developer can re-create the function without seeing any of the 
existing source code.
* Each parameter in a function must have an independent, clear functionality. If two parameters are derivable from 
one another, you are doing something wrong. This also applies for the function's return variables.
* Minimise the number of data types a parameter can be and use common sense. For example, a parameter that can be 
`#!python int` or `#!python None` is reasonable. A parameter that can be `#!python bool` or `#!python float` is not 
reasonable.
* The documentation should update in parallel with the code. Having the documentation as part of the github repository 
makes this easier.

## Run Documentation Locally

Start the documentation locally 

```terminal
mkdocs serve
```

## Docstrings

While not all docstrings are consistent yet, future docstrings follow the rules below:

* The code's functionality must be reproducible from the docstring alone.
* Use [Google's style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html).
* `` `ndarray` `` represents a numpy ndarray and `` `zarray` `` represents a zarr Array.
* `` `zgroup` `` represents a zarr Group.
* Specify datatype of a `ndarray`/`zarray` when applicable. For example, to represent any floating point datatype, 
`` `ndarray[float]` `` or a uint16 by `` `ndarray[uint16]` ``
* Specify the shape of a `ndarray`/`zarray` in brackets when applicable. For example, 
`` `(n_tiles x n_rounds x n_channels_use x 3) ndarray[int32]` ``
* The use of `n_rounds` refers to the number of rounds, including the sequencing and anchor round. So, this is equal to 
`n_seq_rounds + 1`. We label all sequencing rounds `0, 1, 2, 3, ...` and then the anchor round is given the next unused 
integer. Whereas, `n_rounds_use` refers to `#!py len(use_rounds)` which is the total number of sequencing rounds. 
* Channels are slightly different because `use_channels` in the notebook can have channel indices of any positive 
integer value. These represent the sequencing channels. For example, `use_channels = 0, 5, ..., 27`. So, `n_channels` 
refers to size `#!py max(use_channels) + 1`, i.e. the smallest shape that can be indexed by `use_channels`. Whereas, 
`n_channels_use` means `#!py len(use_channels)` such that `0` represents `#!py use_channels[0]` etc. Note that neither 
of these definitions includes the dapi channel/anchor channel[^1], which can be found at `#!py nb.basic_info.dapi_channel` 
and `#!py nb.basic_info.anchor_channel` respectively.

[^1]:
    The anchor channel can also be a sequencing channel.
