import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ezllm",
    version="0.1.0",
    author="José icaro",
    description="Biblioteca super fácil e otimizada para treinamento de redes neurais usando NumPy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Icarogamer2441/ezllm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
) 