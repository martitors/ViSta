# ViSta — Visibility Stacking for Interferometric Data

**ViSta** is a visibility-domain stacking pipeline for interferometric radio and
(sub-)millimeter observations.  
It combines multiple datasets directly in the **uv plane**, after transforming
each visibility dataset into a common rest-frame and phase center.

This approach enhances faint signals, reduces noise, and improves the final
image thanks to the expanded uv coverage from multiple observations.

ViSta is flexible and can stack data with different array configurations,
spectral setups, and sources both on and off the phase center.

---

## Installation

ViSta **must be installed inside a CASA >5 environment**, since it requires
`casatools` and `casatasks`.

Clone and install:

```bash
git clone https://github.com/martitors/ViSta.git
cd ViSta
$CASA_ROOT/bin/python3 -m pip install .
```

Optional (for UVMultiFit-based imaging):

```bash
$CASA_ROOT/bin/python3 -m pip install uvmultifit
```

---

##  Quick Example — Using `run()`

Prepare an input file `sources.txt`:

```
Source1.ms z1  ra1  dec1
Source2.ms z2  ra2  dec2
```

Run ViSta with automatic full pipeline:

```python
from vista import ViSta

# load the pipeline
v = ViSta("sources.txt")
v.restframing()
# run the full pipeline:
# restframing → centering → rebinning → stacking → imaging
v.run(central_freq=...)                
```
---

## Full Example — Manual Step-by-Step Pipeline

You may also run each step independently:

```python
from vista import ViSta

v = ViSta("sources.txt")

# 1) restframe the visibilities 
v.restframing()

# 2) recenter to the phase center and allign the dataset
v.centering()

# 3) rebin to a common frequency grid
v.rebinning(central_freq=230e9)

# 4) stack all MS into a single dataset
v.stacking()

# 5a) imaging with tclean
from vista.imaging_methods import imaging_tclean
imaging_tclean(v.concat_file, imagename="stacked_image")

# 5b) or with uvmultifit (requires time averaging)
from vista.imaging_methods import imaging_uvmultifit
v.averaging(v.concat_file, output_ms="stacked.avg.ms")
imaging_uvmultifit("stacked.avg.ms")
```

---
