import setuptools
from setuptools import setup

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='transformers_keras_dataloader',
    version='0.0.4',
    author="Pratik Deoolwadikar",
    author_email="pratik.deoolwadikar@gmail.com",
    description="Transformers Keras Dataloader provides an EmbeddingDataLoader class, a subclass of keras.utils.Sequence which enables real-time embedding generation from pretrained transformer models while feeding it to your Keras model via batches.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url = 'https://github.com/pratikdk/transformers_keras_dataloader',
    packages=setuptools.find_packages(),
    package_data={'transformers_keras_dataloader': ['config/*.json']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=required,
    keywords = ['transformers', 'embedding', 'dataloader', 'generator', 'huggingface', 'attention'],
)