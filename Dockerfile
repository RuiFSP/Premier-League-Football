# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install pipenv
RUN pip install pipenv

# Create a directory and set it as the working directory
WORKDIR /app

# Copy requirements and install them
COPY ["Pipfile", "Pipfile.lock", "./"]

# Install only production dependencies
RUN pipenv install --system --deploy

# Copy the rest of the files while maintaining the folder structure
COPY predict.py /app/
COPY models/football_model.pkl /app/models/
COPY data/processed/teams_stats_2024.csv /app/data/processed/
COPY scripts/teams_data.py /app/scripts/

EXPOSE 9696

# Run it with Gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]