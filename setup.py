from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='transformers_keras_dataloader',
    version='0.1',
    author="Pratik Deoolwadikar",
    author_email="pratik.deoolwadikar@gmail.com",
    description="Transformers Keras Dataloader provides an EmbeddingDataLoader class, a subclass of keras.utils.Sequence which enables real-time embedding generation from pretrained transformer models while feeding it to your Keras model via batches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License, Version 2.0",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
)