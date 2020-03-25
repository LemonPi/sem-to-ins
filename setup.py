from setuptools import setup

setup(
    name='Semantic to instance segmentation',
    version='0.1.0',
    packages=['sem_to_ins'],
    url='https://github.com/LemonPi/sem-to-ins',
    license='MIT',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='Semantic segmentation masks to instance segmentation masks',
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'opencv-python',
        'matplotlib',
    ],
    tests_require=[
        'pytest'
    ]
)
