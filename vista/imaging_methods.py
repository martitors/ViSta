# ============================================================
#  IMAGING TECHNIQUES: TCLEAN AND UVMULTIFIT
# ============================================================

# This file contains two imaging pipelines:
#   1. Standard CLEAN imaging using CASA tclean
#   2. Visibility-domain fitting using uvmultifit


# ============================================================
# 1. STANDARD IMAGING WITH CASA TCLEAN
# ============================================================

def imaging_tclean(
    vis,
    imagename="image_tclean",
    weighting="natural",
    cell="0.1arcsec",
    niter=0,
    **kwargs
):
    """
    Flexible wrapper for CASA tclean.

    Default parameters:
        weighting = "natural"
        cell      = "0.1arcsec"
        niter     = 0   (dirty map)

    Additional CASA tclean parameters can be passed via **kwargs.
    """

    from casatasks import tclean

    print(f"[TCLEAN] Imaging {vis} → {imagename}")
    print(f"[TCLEAN] Default parameters: weighting={weighting}, cell={cell}, niter={niter}")

    tclean(
        vis=vis,
        imagename=imagename,
        weighting=weighting,
        cell=cell,
        niter=niter,
        **kwargs
    )

    print("[TCLEAN] Done.")
    return imagename



# ============================================================
# 2. VISIBILITY FITTING WITH UVMULTIFIT
# ============================================================

def imaging_uvmultifit(
    vis,
    model=["delta"],
    spw="0",
    column="data",
    var=['0.0', '0.0', 'p[0]'],
    p_ini=[1.0],
    bounds=[[0.0001, 1000.0]],
    output="uvmultifit_output.dat",
    OneFitPerChannel=False,
    cov_return=True
):
    """
    Direct wrapper for uvmultifit using OneFitPerChannel=True.

    Default (delta model):
        dx = 0.0     (fixed)
        dy = 0.0     (fixed)
        flux = p[0]  (only free parameter)

    Inputs:
        vis    : str     - Measurement Set path
        model  : list    - e.g. ["delta"], ["Gaussian"]
        spw    : str     - spectral window selection
        column : str     - data column to use
        var    : list    - parameter expressions for uvmultifit
        p_ini  : list    - initial guess for free parameters
        bounds : list    - bounds for free parameters
        output : str     - output filename

    Output:
        Visibility fit results saved to 'output'.
    """

    try:
        import uvmultifit as uvm
    except ImportError:
        raise ImportError(
        "[ViSta] ERROR: UVMultiFit is not installed inside this CASA environment. "
        "Install it via: pip install uvmultifit or following the GitHub INSTALL guide."
    )


    print(f"[UVMULTIFIT] Fitting {vis}")
    print(f"[UVMULTIFIT] model  = {model}")
    print(f"[UVMULTIFIT] spw    = {spw}")
    print(f"[UVMULTIFIT] column = {column}")
    print(f"[UVMULTIFIT] var    = {var}")
    print(f"[UVMULTIFIT] p_ini  = {p_ini}")
    print(f"[UVMULTIFIT] bounds = {bounds}")
    print(f"[UVMULTIFIT] output = {output}")


    result = uvm.uvmultifit(
        vis=vis,
        model=model,
        spw=spw,
        column=column,
        var=var,
        p_ini=p_ini,
        bounds=bounds,
        OneFitPerChannel=OneFitPerChannel,
        cov_return=cov_return,
        outfile=output
    )

    print("[UVMULTIFIT] Done.")
    return result

