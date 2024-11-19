from setuptools import setup, find_packages

setup(
    name='llms',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'openai',
        'tiktoken',
        'transformers',
        'torch',
        'requests',
        # Add other dependencies as needed
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A custom library for interacting with various Language Models',
    url='https://github.com/yourusername/llms',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
