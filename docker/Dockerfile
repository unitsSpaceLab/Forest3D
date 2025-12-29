# Forest3D Docker image with GDAL, Blender, and Gazebo Harmonic
# Build: docker build -t forest3d -f docker/Dockerfile .
# Run:   docker run -v $(pwd):/workspace forest3d --help
# Run with GUI: docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(pwd):/workspace forest3d launch

FROM ubuntu:22.04

LABEL maintainer="AI4Forest <khalid.bourr@gmail.com>"
LABEL description="Forest3D - Terrain and forest generation for Gazebo simulation"
LABEL version="0.1.0"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Blender version to install
ARG BLENDER_VERSION=4.2.3
ARG BLENDER_URL=https://download.blender.org/release/Blender4.2/blender-4.2.3-linux-x64.tar.xz

# Install system dependencies including Python, GDAL, Blender requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    gdal-bin \
    libgdal-dev \
    python3-gdal \
    wget \
    xz-utils \
    gnupg \
    lsb-release \
    curl \
    software-properties-common \
    # Blender dependencies
    libxi6 \
    libxxf86vm1 \
    libxfixes3 \
    libxrender1 \
    libgl1 \
    libxkbcommon0 \
    # Gazebo/GUI dependencies
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libxcb-xinerama0 \
    libxcb-cursor0 \
    libxkbcommon-x11-0 \
    libdbus-1-3 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Add OSRF repository for Gazebo Harmonic
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" > /etc/apt/sources.list.d/gazebo-stable.list

# Install Gazebo Harmonic
RUN apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    && rm -rf /var/lib/apt/lists/*

# Download and install Blender
RUN mkdir -p /opt/blender && \
    wget -q --show-progress -O /tmp/blender.tar.xz ${BLENDER_URL} && \
    tar -xf /tmp/blender.tar.xz -C /opt/blender --strip-components=1 && \
    rm /tmp/blender.tar.xz && \
    ln -s /opt/blender/blender /usr/local/bin/blender

# Set GDAL environment variables
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Set Blender path for Forest3D
ENV FOREST3D_BLENDER_PATH=/opt/blender/blender

# Set Gazebo model path
ENV GZ_SIM_RESOURCE_PATH=/workspace/models

# Create non-root user for security
RUN useradd -m -s /bin/bash forest3d
WORKDIR /home/forest3d

# Copy and install Forest3D
COPY --chown=forest3d:forest3d . /home/forest3d/forest3d-src

# Upgrade pip and install Forest3D
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir /home/forest3d/forest3d-src && \
    pip3 install --no-cache-dir pygdal==$(gdal-config --version).* || true

USER forest3d

# Default working directory for mounting projects
WORKDIR /workspace

# Set environment for mounted projects
ENV FOREST3D_BASE_PATH=/workspace

ENTRYPOINT ["forest3d"]
CMD ["--help"]
