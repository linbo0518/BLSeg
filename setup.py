from setuptools import setup, find_packages

VERSION = "20200521"
README = open("README.md").read()
REQUIREMENTS = ["torch", "numpy"]

setup(
    name="blseg",
    version=VERSION,
    author="Leon Lin",
    author_email="linbo0518@gmail.com",
    url="https://github.com/linbo0518/BLSeg",
    description="linbo0518@gmail.com",
    long_description=README,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)