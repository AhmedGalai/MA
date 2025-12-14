#!/bin/bash

echo "========================================="
echo "Stopping UxPlay Docker Container"
echo "========================================="
echo ""

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "[ERROR] docker-compose is not installed."
    exit 1
fi

# Stop the UxPlay container
echo "[INFO] Stopping UxPlay container..."
docker-compose down uxplay

echo ""
echo "========================================="
echo "UxPlay has been stopped"
echo "========================================="
echo ""
echo "The container has been stopped and removed."
echo "The frames directory and its contents remain."
echo ""
echo "To start UxPlay again:"
echo "  ./start_uxplay.sh"
echo ""
