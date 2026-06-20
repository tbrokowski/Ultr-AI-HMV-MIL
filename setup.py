from setuptools import setup, find_packages

# Read from requirements.txt
# with open('requirements.txt') as f:
#     required = f.read().splitlines()
#     install_requires = required

setup(
    name='hmv-mil',
    version='0.1.0',
    description='HMV-MIL training code for lung ultrasound TB classification',
    long_description='HMV-MIL training code for lung ultrasound TB classification',
    long_description_content_type='text/markdown',
    packages=find_packages(),
    # install_requires=install_requires,
)