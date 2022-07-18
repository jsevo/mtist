import setuptools

with open("readme.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mtist",
    version="1.0",
    author=["Grant Hussey", "Jonas Schluter"],
    author_email=["grant.hussey@nyulangone.org", "jonas.schluter@nyulangone.org"],
    description="MTIST: A platform to benchmark ecosystem inference algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jsevo/mtist",
    # project_urls={
    #     "Bug Tracker": "https://github.com/pypa/sampleproject/issues",
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": ""},
    packages=["mtist"],
    # packages=setuptools.find_packages(where=""),
    # matplotlib numpy pandas scipy scikit-learn seaborn
    python_requires=">=3.6",
)
