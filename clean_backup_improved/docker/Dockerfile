FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build tools
    build-essential \
    cmake \
    git \
    pkg-config \
    # UxPlay dependencies
    libavahi-compat-libdnssd-dev \
    libplist-dev \
    libssl-dev \
    # GStreamer
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    # Avahi for mDNS/AirPlay discovery
    avahi-daemon \
    avahi-utils \
    # Python 3
    python3 \
    python3-pip \
    python3-dev \
    # Additional utilities
    wget \
    curl \
    vim \
    net-tools \
    && rm -rf /var/lib/apt/lists/*

# Build and install UxPlay from source
WORKDIR /tmp
RUN git clone https://github.com/FDH2/UxPlay.git && \
    cd UxPlay && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig && \
    cd / && \
    rm -rf /tmp/UxPlay

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r /app/requirements.txt

# Set up application directory
WORKDIR /app

# Copy application files
COPY uxplay_module.py /app/
COPY config.py /app/
COPY run_uxplay_integration.py /app/ 2>/dev/null || true

# Create frames directory for UxPlay output
RUN mkdir -p /app/frames

# Configure avahi-daemon to work in Docker
RUN mkdir -p /var/run/avahi-daemon && \
    sed -i 's/#enable-dbus=yes/enable-dbus=no/' /etc/avahi/avahi-daemon.conf && \
    sed -i 's/rlimit-nproc=3/rlimit-nproc=8/' /etc/avahi/avahi-daemon.conf

# Expose AirPlay ports
# 7000: AirPlay control
# 7001: AirPlay data
# 6000-6001: Additional AirPlay ports
# 5353: mDNS (Bonjour/Avahi)
EXPOSE 7000 7001 6000 6001 5353/udp

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV MAIN_API_HOST=192.168.178.68
ENV MAIN_API_PORT=8000

# Start script
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
