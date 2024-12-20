## Pipeline crash

If the coppafish pipeline is crashing, first read the error message. If there is a suggestion about how to fix the
issue in the config, try changing the config variable and run the pipeline again. If the suggestion does not make sense
to you, feel free to reach out to the developers for help or
[create an issue](https://github.com/paulshuker/coppafish/issues/new?assignees=&labels=&projects=&template=bug.md&title=)
on GitHub!

## Memory crash at OMP

Try reducing `subset_pixels` in the OMP config. This will cause OMP to compute on fewer pixels at time. It has a
minimal effect on compute times, but will lower the RAM/VRAM usage. By default, the subset pixel number is found on
the fly based on the PC's hardware, you can see this by looking at the pipeline.log file and searching for
`subset_pixels`.
