from setuptools import setup

setup(
    name='time_series_cross_validation',
    version='1.0.4',
    description='Library for cross-validating time series',
    long_description = 'Library for cross-validating time series',
    author='Gabriel Nuernberg Biazoto',
    author_email='biazotogabriel@gmail.com',
    url='https://github.com/biazotogabriel/deep_tracking',
    packages=['time_series_cross_validation'],
    license = 'MIT',
    keywords = 'deep time series cross validation data science',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas==1.4.4',
        'matplotlib==3.5.2'
    ],
)
