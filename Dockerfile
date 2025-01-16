# Use the Ultralytics Docker image with GPU support as the base image
FROM ultralytics/ultralytics:latest

# Set the working directory inside the container
WORKDIR /workspace

# Copy your requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Create a directory for data to be mounted as a volume (if needed)
RUN mkdir -p /workspace/data

# Remove the CMD that starts Jupyter Notebook server
# Override the CMD to prevent the container from exiting
CMD ["sleep", "infinity"]