import setuptools

setuptools.setup(
    name="robotini-ddpg",
    version="0.1.0",
    description="Deep deterministic policy gradient agents for the Robotini racing simulator",
    # long_description=readmefile_contents,
    # long_description_content_type="text/markdown",
    author="Matias Lindgren",
    author_email="matias.lindgren@iki.fi",
    url="https://github.com/matiaslindgren/robotini-ddpg",
    license="MIT",
    python_requires=">= 3.9.*",
    packages=[
        "robotini_ddpg",
        "robotini_ddpg.model",
        "robotini_ddpg.simulator",
        "robotini_ddpg.monitor",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
