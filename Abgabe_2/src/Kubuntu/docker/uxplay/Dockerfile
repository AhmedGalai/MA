FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libavahi-compat-libdnssd-dev \
    libplist-dev \
    libssl-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    && rm -rf /var/lib/apt/lists/*

# Clone and build UxPlay
WORKDIR /tmp
RUN git clone https://github.com/FDH2/UxPlay.git && \
    cd UxPlay && \
    mkdir build && cd build && \
    cmake .. && \
    make && \
    make install && \
    ldconfig && \
    rm -rf /tmp/UxPlay

# Create output directory
RUN mkdir -p /frames

# Startup script
COPY start-uxplay.sh /start-uxplay.sh
RUN chmod +x /start-uxplay.sh

EXPOSE 7000 7001 7100

CMD ["/start-uxplay.sh"]
