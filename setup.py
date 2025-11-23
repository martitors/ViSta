from setuptools import setup, find_packages
import importlib
import sys

def casa_modules_available():
    """
    Check if casatools and casatasks are already installed.
    """
    try:
        importlib.import_module("casatools")
        importlib.import_module("casatasks")
        return True
    except ImportError:
        return False


# ---------------------------------------------------------
# REQUIREMENTS
# ---------------------------------------------------------
base_requirements = [
    "numpy",
    "astropy",
    "pyvirtualdisplay"
]

# ---------------------------------------------------------
# BLOCK INSTALLATION OUTSIDE CASA
# ---------------------------------------------------------
if not casa_modules_available():
    sys.stderr.write(
        "\nERROR: ViSta can only be installed inside a CASA environment.\n"
        "Required modules 'casatools' and 'casatasks' were not found.\n\n"
    )
    sys.exit(1)

# If we are inside CASA, no need to install CASA packages
install_requires = base_requirements

# ---------------------------------------------------------
# PACKAGE SETUP
# ---------------------------------------------------------
setup(
    name="ViSta",
    version="0.5.0",
    packages=find_packages(),
    python_requires=">=3.8",
    description="ViSta — Virtual Stacking Pipeline for ALMA Measurement Sets",
    install_requires=install_requires,
)

