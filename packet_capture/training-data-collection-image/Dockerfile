# Use the watchdog as a base layer
FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.9.15 as watchdog

# Use Debian-based Node.js image instead of Alpine
FROM --platform=${TARGETPLATFORM:-linux/amd64} node:18-slim as ship

# Set build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Copy watchdog
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Install necessary packages on Debian
RUN apt-get update && apt-get install -y --no-install-recommends \
     build-essential  g++ \
      #    python3 python3-pip  python3-dev libjsoncpp-dev  \
    && rm -rf /var/lib/apt/lists/*


# Secure the environment and set permissions for root-only modifications
RUN echo 'export LD_PRELOAD=/usr/src/app/libmessage_training.so' > /etc/profile.d/preload.sh && \
    chmod 644 /etc/profile.d/preload.sh && \
    chown root:root /etc/profile.d/preload.sh

# Set working directory
WORKDIR /usr/src/app

# Copy source files and Makefile into the container
COPY common_entities_training/ ./

#RUN find / -name "json.h"

ENV PYTHONPATH="/usr/src/app"

# Build the shared library
RUN make all

# Ensure the library is owned by root and has correct permissions
RUN chown root:root libmessage_training.so && \
    chmod 755 libmessage_training.so




#ENV LD_PRELOAD=/usr/src/app/libmessage_training.so

# Healthcheck
HEALTHCHECK --interval=6s CMD [ -e /tmp/.lock ] || exit 1



# Set the command to start fwatchdog
#CMD ["fwatchdog"]

