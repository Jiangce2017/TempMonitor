import os
from setuptools import find_packages, setup

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_version():
    with open(os.path.join(_CURRENT_DIR, "src/TempMonitor/__version__.py")) as file:
        for line in file:
            if line.startswith("__version__"):
                return line[line.find("=") + 1:].strip(' \'"\n')
        raise ValueError('`__version__` not defined in `TempMonitor/__version__.py`')

__version__ = get_version()

if __name__=='__main__':
    setup(
        name="TempMonitor",
        version=__version__,
        description="Derive the inside temperature from the surface temperature for monitoring additive manufacturing process",
        author="Jiangce Chen",
        author_email="jiangcechen@gmail.com",
        long_description=open(os.path.join(_CURRENT_DIR, "README.md")).read(),
        long_description_content_type='text/markdown',
        url="https://github.com/Jiangce2017/TempMonitor",
        license="MIT",
    )