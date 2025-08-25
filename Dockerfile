# Start from an official Python base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /code

# Create a non-root user for security
RUN useradd -m appuser
USER appuser

# Copy the requirements file first to leverage Docker's caching
COPY --chown=appuser:appuser requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# CRITICAL STEP: Install the browser binaries needed by Crawl4ai/Playwright
# The --with-deps flag also installs necessary system libraries.
RUN python -m playwright install --with-deps chromium

# Copy the rest of your application code into the container
COPY --chown=appuser:appuser . .

# Expose the port that Hugging Face Spaces expects
EXPOSE 7860

# The command to run your FastAPI server when the container starts
CMD ["uvicorn", "mcp_server:app", "--host", "0.0.0.0", "--port", "7860"]