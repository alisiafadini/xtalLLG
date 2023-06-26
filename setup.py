DESCRIPTION      =   "Maximum Likelihood target implementation for protein crystallography in Pytorch"
LONG_DESCRIPTION = """
Work in progress
"""

try:
	from setuptools import setup, find_packages

except ImportError:
	from disutils.core import setup

    
setup(
    name='xtalLLG',
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author='Alisia Fadini',
    author_email="af840@cam.ac.uk",
    install_requires=["reciprocalspaceship", "matplotlib", "gemmi", "SFcalculator_torch"],
    packages=find_packages(),
)
