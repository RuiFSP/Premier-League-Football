# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install pipenv
RUN pip install pipenv

# Create a directory and set it as the working directory
WORKDIR /app

# Install all the dependencies form the Pipenv file
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install the dependencies
RUN pipenv install --system --deploy

# Copy the rest of the files
COPY ["predict.py", "models/football_model.pkl", "./"]

EXPOSE 9696

# Run it with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]