# setup.py

from setuptools import setup, find_packages

setup(
    name='quantum_ml_library',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'qiskit',
    ],
    python_requires='>=3.6',
    author='Your Name',
    author_email='you@example.com',
    url='https://github.com/yourusername/quantum_ml_library',
    description='A quantum machine learning library for high-tech projects',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
