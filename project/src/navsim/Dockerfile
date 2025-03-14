# Use a Miniconda base image for easy Conda environment management
FROM continuumio/miniconda3:latest

# Install git for cloning the repository
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /navsim_workspace

# Clone the NAVSIM repository
RUN git clone https://github.com/autonomousvision/navsim.git

# Create required directory structure for datasets and experiments
RUN mkdir -p dataset/maps \
    dataset/navsim_logs/{test,trainval,private_test_e2e,mini} \
    dataset/sensor_blobs/{test,trainval,private_test_e2e,mini} \
    dataset/synthetic_scenes/{scene_pickles,sythetic_sensor} \
    exp

# Create the conda environment from the provided environment.yml in the navsim folder
RUN conda env create --file navsim/environment.yml

# Switch to using the newly created "navsim" environment for all subsequent commands
SHELL ["conda", "run", "-n", "navsim", "/bin/bash", "-c"]

# Install the navsim-devkit in editable mode
RUN pip install -e navsim

# Set required environment variables (modify if your mounted paths differ)
ENV NUPLAN_MAP_VERSION="nuplan-maps-v1.0" \
    NUPLAN_MAPS_ROOT="/navsim_workspace/dataset/maps" \
    NAVSIM_EXP_ROOT="/navsim_workspace/exp" \
    NAVSIM_DEVKIT_ROOT="/navsim_workspace/navsim" \
    OPENSCENE_DATA_ROOT="/navsim_workspace/dataset"

# Copy the entrypoint script into the container and make it executable
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint script and default command (bash shell)
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]