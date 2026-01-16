from setuptools import setup, find_packages

setup(
    name="ai-resume-screening-system",
    version="1.0.0",
    description="Production-ready AI resume screening system",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip() 
        for line in open('requirements.txt').readlines()
        if line.strip() and not line.startswith('#')
    ],
    python_requires=">=3.8",
    entry_points={
        'console_scripts': [
            'resume-screening=src.api.main:main',
        ],
    },
)
