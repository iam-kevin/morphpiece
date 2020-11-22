# Attempting to install
#  using `python-semantic-release`
try:
    import sys
    from semantic_release import setup_hook
    setup_hook(sys.argv)
except ImportError:
    pass

import os
from setuptools import setup
from morphpiece import __version__ as VERSION

# Utility function for reading files
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name = "morphpiece",
    version = VERSION,
    author = "Kevin James",
    author_email = "kevin@inspiredideas.com",
    description = "A Semi-supervised Learning Text Tokenizer",
    license = "LGPL",
    keywords = "nlp morphpiece tokenizer semi supervised",
    url = "https://github.com/iam-kevin/morphpiece",
    packages=['morphpiece', 'tests'],
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
    ],
)
