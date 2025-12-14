#!/bin/bash

# GStreamer pipeline: write frames to shared directory
# videorate ensures constant frame rate
# multifilesink overwrites same file (latest.jpg)
VIDEOSINK="videorate ! video/x-raw,framerate=30/1 ! videoconvert ! jpegenc quality=90 ! multifilesink location=/frames/latest.jpg max-files=1"

echo "Starting UxPlay with GStreamer output to /frames/latest.jpg"
echo "Device name: Kubuntu Backend"
echo "Waiting for AirPlay connection..."

uxplay -n "Kubuntu Backend" \
       -vs "$VIDEOSINK" \
       -reset 3 \
       -vsync no \
       -fps 30
