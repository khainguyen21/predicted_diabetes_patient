FROM ubuntu:latest

WORKDIR /src

# Install necessary dependencies
RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv

# Create and set up a virtual environment which contains its own Python interpreter
# The -m flag means "run a Python module as a script."
# venv is a built-in Python module that creates virtual environments.
RUN python3 -m venv myenv

RUN myenv/bin/pip install pandas scikit-learn

# Ensure the virtual environment is used by default
# This line adds /src/myenv/bin to the beginning of the PATH.
# So now, when you type python3 or pip, the system first looks in
# /src/myenv/bin before checking the other locations.
ENV PATH="/src/myenv/bin:$PATH"

# Copy the script and dataset into the container
COPY diabetes_rs.py ./diabetes_rs.py
COPY diabetes.csv ./diabetes.csv

# Default command: Open a shell
