The coppafish pipeline is separated into distinct sections. Some of these are for image pre-processing
([scale](#scale), [extract](#extract), [filter](#filter)), image alignment ([register](#register), [stitch](#stitch))
and spot detection/gene calling ([find spots](#find-spots), [call spots](#call-spots),
[orthogonal matching pursuit](#orthogonal-matching-pursuit)). Below, each section is given in the order a coppafish
pipeline runs in.

## Extract

Save all raw data again at the `tile_dir` in the `extract` config section. Coppafish does this for:

* file compression support.
* saving raw data in a universal format that can then be used by multiple versions of our software.
* optimised data retrieval speeds. The default file type is using [zarr](https://zarr.readthedocs.io/) arrays, but we
also support saving as uncompressed numpy arrays by setting `file_type` to `.npy` in the extract config section.

Extract also saves metadata inside of the `tile_dir` directory if the raw files are ND2 format.

Extract takes $\textsf{n_pixels}\times1.2\times10^{-8}$ minutes to complete from raw npy files on a local NVMe SSD and
$\textsf{n_pixels}\times7\times10^{-7}$ minutes to complete from raw ND2 files with 100MB/s reading speed, where
$\textsf{n_pixels}$ is the total number of pixels in your dataset[^1].

## Filter

All images are filtered to help minimise scattering of light (bright points will appear as cones initially, hence the
name "Point Spread Function") and emphasis spots. The parts to this are:

* calculating a Point Spread Function (PSF) using “good” spot shapes which is used to apply a
<a href="https://en.wikipedia.org/wiki/Wiener_deconvolution" target="_blank">Wiener filtering</a> on every image if
`deconvolve` in the `filter` config is set to true (default). This is to reduce image blur caused by light scattering.
* applying a smoothing kernel (this is just an un-weighted average) to every image by setting `r_smooth` in the
`filter` config section. By default, this is not applied.
* a difference of two <a href=https://en.wikipedia.org/wiki/Hann_function target="_blank">Hannings</a> 2D kernel is
applied to every image that is not a DAPI. By default, this is not applied. If it is a DAPI, instead apply a 2D top hat
filter (which is just a 2D top hat kernel) of size `r_dapi` if it is set manually to a number in the config. By
default, this is not applied.

After filtering is applied, the images are scaled by a computed scale factor and then saved in `uint16` format again.
By default, only the Wiener deconvolve is applied as this is expected to be near optimal.

Filter takes $\textsf{n_pixels}\times4\times10^{-8}$ minutes.

## Find spots

Point clouds (a series of spot x, y, and z locations) are generated for each filtered image. These are found by
detecting local maxima in image intensity around the rough spot size (specified by config variables `radius_xy` and
`radius_z` in the `find_spots` section). If two local maxima are the same value and in the same spot region, then one
is chosen at random. Warnings and errors are raised if there are too few spots detected in a round/channel, these can
be customised, see `find_spots` section in the
<a href="https://github.com/reillytilbury/coppafish/blob/alpha/coppafish/setup/settings.default.ini" target="_blank">
config</a> default file for variable names.

Find spots takes $\textsf{n_pixels}\times3\times10^{-9}$ minutes.

## Register

## Stitch

## Call spots

## Orthogonal Matching Pursuit

Orthogonal Matching Pursuit (OMP) is the most sophisticated gene calling method used by coppafish, allowing for
overlapping genes to be detected. It is an iterative,
<a href="https://en.wikipedia.org/wiki/Greedy_algorithm" target="_blank">greedy algorithm</a> that runs on individual
pixels of the images. At each OMP iteration, a new gene is assigned to the pixel. OMP is also self-correcting.
"Orthogonal" refers to how OMP will re-compute its gene contributions after every iteration by least squares.
Background genes[^2] are considered valid genes in OMP. The iterations stop if:

* `max_genes` in the `omp` config section is reached.
* assigning the next best gene to the pixel does not have a dot product score above `dp_thresh` in the `omp` config.
The dot product score is a dot product of the residual pixel intensity in every sequencing round/channel (known as its
colour) with the normalised bled codes (see [call spots](#call-spots)).

Sometimes, when a gene is chosen by OMP, a very strong residual pixel intensity can be produced when the selected gene
is subtracted from the pixel colour. To protect against this, `weight_coef_fit` can be set to true and weighting
parameter `alpha` ($\alpha$) can be set in the `omp` config. When $\alpha>0$, round/channel pixel intensities largely
contributed to by previous genes are fitted with less importance in the next iteration(s). In other words, $\alpha$
will try soften any large outlier pixel intensities.

<!-- TODO: Should expand more on the OMP gene scoring here -->
After a pixel map of gene coefficients is found through OMP on many image pixels, spots are detected as local
coefficient maxima (similar to [find spots](#find-spots)). Spots are scored by a weighted average around a small local
region of the spot where the spot is expressed most strongly. The coefficients are weighted with the mean spot
intensity normalised to have a maximum of 1. The mean spot is computed on tile `nb.basic_info.use_tiles[0]` by taking
the average of many well-isolated spots. The scoring is controlled by config parameters `shape_sign_thresh` and
`high_coef_bias`. Low scores are deleted by OMP when they are below the `score_threshold`.

Since OMP is sensitive to the many steps before, it can be difficult to optimise. This is why [call spots](#call-spots)
is part of the gene calling pipeline, known for its simpler and more intuitive method. A good sanity check is to see if
OMP and call spots have similar gene reads. But, you should expect to see more gene calls made by OMP compared to call
spots.

OMP takes $\textsf{n_pixels}\times2\times10^{-7}$ minutes for pytorch CPU and $\textsf{n_pixels}\times1.4\times10^{-7}$
for pytorch GPU.


[^1]:
    All time estimations are rough and made using CPU pytorch with an Intel i9-13900K @ 5.500GHz unless otherwise
    stated.
[^2]:
    Background genes refer to constant pixel intensity across all sequencing rounds in one channel. This is an
    indicator of an anomalous fluorescing feature that is not a spot. No spot codes are made to be the same channel in
    all rounds so they are not mistaken with background fluorescence.
