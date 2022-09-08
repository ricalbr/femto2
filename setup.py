from setuptools import find_packages, setup

setup(
        name='femto',
        description='Python suite for the design of femtosecond laser-written circuit.',
        version='1.2',
        packages=find_packages(include=['femto',
                                        'compiler',
                                        'compiler.*',
                                        'objects',
                                        'objects.*']),
        install_requires=[
                'matplotlib',
                'numpy',
                'pandas',
                'plotly',
                'scipy',
        ],
)
