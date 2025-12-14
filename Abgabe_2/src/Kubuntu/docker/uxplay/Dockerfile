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

# Create startup script directly in Dockerfile (avoids line ending issues)
RUN echo '#!/bin/bash' > /start-uxplay.sh && \
    echo '' >> /start-uxplay.sh && \
    echo '# GStreamer pipeline: write frames to shared directory' >> /start-uxplay.sh && \
    echo '# videorate ensures constant frame rate' >> /start-uxplay.sh && \
    echo '# multifilesink overwrites same file (latest.jpg)' >> /start-uxplay.sh && \
    echo 'VIDEOSINK="videorate ! video/x-raw,framerate=30/1 ! videoconvert ! jpegenc quality=90 ! multifilesink location=/frames/latest.jpg max-files=1"' >> /start-uxplay.sh && \
    echo '' >> /start-uxplay.sh && \
    echo 'echo "Starting UxPlay with GStreamer output to /frames/latest.jpg"' >> /start-uxplay.sh && \
    echo 'echo "Device name: Kubuntu Backend"' >> /start-uxplay.sh && \
    echo 'echo "Waiting for AirPlay connection..."' >> /start-uxplay.sh && \
    echo '' >> /start-uxplay.sh && \
    echo 'uxplay -n "Kubuntu Backend" \' >> /start-uxplay.sh && \
    echo '       -vs "$VIDEOSINK" \' >> /start-uxplay.sh && \
    echo '       -reset 3 \' >> /start-uxplay.sh && \
    echo '       -vsync no \' >> /start-uxplay.sh && \
    echo '       -fps 30' >> /start-uxplay.sh && \
    chmod +x /start-uxplay.sh

EXPOSE 7000 7001 7100

CMD ["/start-uxplay.sh"]
