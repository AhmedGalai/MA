FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
  build-essential cmake git \
  libavahi-compat-libdnssd-dev libplist-dev libssl-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-base gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad gstreamer1.0-libav gstreamer1.0-tools \
  avahi-daemon dbus \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone https://github.com/FDH2/UxPlay.git && \
  cd UxPlay && mkdir build && cd build && \
  cmake .. && make && make install && ldconfig && \
  rm -rf /tmp/UxPlay

RUN mkdir -p /frames /var/run/dbus

EXPOSE 7000 7001 7100 6000 6001

CMD mkdir -p /var/run/dbus && (dbus-daemon --system --fork || true) && (avahi-daemon --daemonize --no-chroot || true) && sleep 2 && uxplay -n "Kubuntu Backend" -vs "videorate ! video/x-raw,framerate=30/1 ! videoconvert ! jpegenc quality=90 ! multifilesink location=/frames/latest.jpg max-files=1" -reset 3 -vsync no -fps 30