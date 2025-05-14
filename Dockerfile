# Use a slim version of Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /code

# Copy the requirements file to the container and install dependencies
COPY ./app/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r /code/requirements.txt

# Copy the application code to the container
COPY ./app /code/app

# Copy the model and tokenizer configuration files into the container
COPY ./app/saved_model.keras /code/saved_model.keras
COPY ./tokenizer_config.json /code/tokenizer_config.json

# Expose port 8000 for FastAPI
EXPOSE 8000

# Command to run the application using Uvicorn
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8000"]
