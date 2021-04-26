from setuptools import setup

setup(
    name='mtist',
    version='0.1',
    description='Ecosystem simulation for inference comparison',
    url='http://github.com/jsevo/mtist',
    author='Jonas Schluter',
    author_email='jonas.schluter+github@gmail.com',
    license='MIT',
    packages=['mtist'],
    install_requires=[
        'matplotlib',
        'pandas',
        'seaborn',
        'numpy',
        'scipy'
    ],
    zip_safe=False)
