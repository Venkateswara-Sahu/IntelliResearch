from setuptools import setup, find_packages

setup(
    name="intelliresearch",
    version="0.1.0",
    description="Multi-agent research automation system",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "streamlit",
        "pytest",
        "pytest-asyncio",
        "requests",
        "beautifulsoup4",
        "pandas",
    ],
    python_requires=">=3.9",
)
