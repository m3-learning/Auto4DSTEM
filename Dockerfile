# Use an official Miniconda image as a parent image

FROM continuumio/miniconda3
 
# Set the working directory in the container

WORKDIR /usr/src/app
 
# Copy the current directory contents into the container at /usr/src/app

COPY . /usr/src/app
 
# Optionally, if you have a separate environment file (e.g., environment.yml)

# and prefer to use it instead of a requirements.txt for Conda, you would copy

# it into the Docker image and create the Conda environment like this:

# COPY environment.yml /usr/src/app/environment.yml

# RUN conda env create -f environment.yml
 
# Use the requirements.txt to create a Conda environment. Assume requirements.txt is already copied

RUN conda create --name myenv --file requirements.txt
 
# Make RUN commands use the new environment:

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
 
# Ensure the environment is activated each time the container starts.

# This is useful for interactive usage or if you're extending this Dockerfile

# to run a specific application.

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "myenv"]
 
# The following is an example command you might use to run a Python script.

CMD ["python", "your_script.py"]