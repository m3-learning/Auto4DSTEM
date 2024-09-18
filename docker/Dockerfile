# Use an official Python runtime as a parent image
FROM python:3.10-slim

VOLUME /app/data

# Set the working directory in the container to /app
WORKDIR /app


# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app
COPY ./Experimental_label_weights_affine_para /app
COPY ./simulated_label_weights_affine_para  /app
COPY ./Auto4DSTEM_Tutorial_Supplemental_Material_Experimental_4DSTEM_update_051024.ipynb /app
COPY ./Auto4DSTEM_Tutorial_Supplemental_Material_Simulated_4DSTEM_update_051024.ipynb /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# # Make port 80 available to the world outside this container
# EXPOSE 80

# # Define environment variable
# ENV NAME World

# # Run app.py when the container launches
# CMD ["python", "app.py"]

# Install Jupyter
RUN pip install jupyter

# Expose port 8999 for the Jupyter Notebook
EXPOSE 8999

# Run Jupyter Notebook
# --NotebookApp.token='' disables token auth, not recommended for production environments
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--port=8999", "--no-browser", "--allow-root"]