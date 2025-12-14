#!/bin/bash

echo "========================================="
echo "Starting UxPlay Docker Container"
echo "========================================="
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "[ERROR] Docker is not running. Please start Docker first."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] docker-compose is not installed. Please install it first."
    exit 1
fi

# Create frames directory if it doesn't exist
echo "[INFO] Creating frames directory..."
mkdir -p frames

# Build the UxPlay container if needed
echo "[INFO] Building UxPlay Docker image (if needed)..."
docker-compose build uxplay

# Start the UxPlay container
echo "[INFO] Starting UxPlay container..."
docker-compose up -d uxplay

# Wait for container to start
sleep 2

# Check status
echo ""
echo "========================================="
echo "Container Status"
echo "========================================="
docker-compose ps uxplay

echo ""
echo "========================================="
echo "UxPlay is now running!"
echo "========================================="
echo ""
echo "To use:"
echo "  1. On your Apple Vision Pro or iOS device:"
echo "     - Open Control Center"
echo "     - Tap Screen Mirroring"
echo "     - Select 'Kubuntu Backend'"
echo ""
echo "  2. Frames will be continuously written to:"
echo "     ./frames/latest.jpg"
echo ""
echo "  3. Use the Debug Viewer or VisionOS app to capture frames"
echo ""
echo "To view logs:"
echo "  docker-compose logs -f uxplay"
echo ""
echo "To stop UxPlay:"
echo "  ./stop_uxplay.sh"
echo ""
