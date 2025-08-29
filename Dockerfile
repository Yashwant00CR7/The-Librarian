# Start from an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Add the user's local bin to the PATH
# This ensures that executables installed by pip (like uvicorn) can be found.
ENV PATH="/home/appuser/.local/bin:${PATH}"

# Copy the requirements file first to leverage Docker's caching
COPY --chown=appuser:appuser requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# CRITICAL STEP (CORRECTED): Install the browser binaries WITHOUT system dependencies
RUN python -m playwright install chromium

# Copy the rest of your application code into the container
COPY --chown=appuser:appuser . .

# Expose the port that Cloud Run will assign (this is for documentation)
EXPOSE 8080

# The command to run your FastAPI server
# This "shell form" correctly interprets the $PORT environment variable.
CMD uvicorn mcp_server:app --host 0.0.0.0 --port $PORT
