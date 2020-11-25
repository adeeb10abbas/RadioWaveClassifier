from setuptools import setup

setup(
    name='modrec',
    author='Geoffrey Mainland',
    author_email='mainland@drexel.edu',
    url='https://github.com/mainland/modrec',
    description='Modulation recognition with Tensorflow',
    long_description='',
    packages=['modrec'],
    version='0.1',
    install_requires=['h5py==2.10.0'
                     ,'image-classifiers==1.0.0'
                     ,'matplotlib==3.3.2'
                     ,'numpy<1.19.0'
                     ,'pandas==1.1.3'
                     ,'tensorflow==2.3.1'
                     ]
)
