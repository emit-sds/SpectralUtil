__version__ = "0.0.2"

# Load CLI first
from .__main__ import cli

# Attach CLI commands
from spectral_util import io
from spectral_util import utils
from spectral_util.io import mosaic
