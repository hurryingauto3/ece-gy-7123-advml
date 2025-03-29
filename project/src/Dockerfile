FROM continuumio/miniconda3:latest

# --- Install system dependencies ---
RUN apt-get update && apt-get install -y \
    git \
    gdal-bin \
    libgdal-dev \
    build-essential \
    python3-dev \
    unzip \
    libgl1-mesa-glx \
&& rm -rf /var/lib/apt/lists/*

# --- Set GDAL environment variables ---
ENV GDAL_CONFIG=/usr/bin/gdal-config \
    CPLUS_INCLUDE_PATH=/usr/include/gdal \
    C_INCLUDE_PATH=/usr/include/gdal

# --- Set working directory ---
WORKDIR /navsim_workspace

# --- Clone the navsim repository and setup directories ---
    RUN git clone https://github.com/autonomousvision/navsim.git && \
    chmod -R +x navsim/download
# --- Create conda environment using environment.yml from the cloned repository ---
RUN conda env create --name navsim --file navsim/environment.yml

# --- Activate the navsim environment for all subsequent commands ---
ENV PATH /opt/conda/envs/navsim/bin:$PATH

# --- Install navsim in editable mode and setup Jupyter with kernel ---
RUN /bin/bash -c "source activate navsim && \
    pip install --no-cache-dir -e navsim && \
    conda install -y jupyter ipykernel notebook ipywidgets && \
    python -m ipykernel install --user --name=navsim"

# --- Set project-specific environment variables ---
# environment variables so you don’t manually do them each time
ENV NUPLAN_MAP_VERSION="nuplan-maps-v1.0" \
    NUPLAN_MAPS_ROOT="/navsim_workspace/dataset/maps" \
    NAVSIM_EXP_ROOT="/navsim_workspace/exp" \
    NAVSIM_DEVKIT_ROOT="/navsim_workspace/navsim" \
    OPENSCENE_DATA_ROOT="/navsim_workspace/dataset" \
    HYDRA_FULL_ERROR=1 \
    JUPYTER_TOKEN=""

# --- Expose the Jupyter Notebook port ---
EXPOSE 8888 

# --- Copy and enable the entrypoint script ---
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

# --- Default command: Launch Jupyter Notebook in the workspace ---
CMD ["jupyter", "notebook", "--notebook-dir=/navsim_workspace", "--ip=0.0.0.0", "--allow-root", "--no-browser"]