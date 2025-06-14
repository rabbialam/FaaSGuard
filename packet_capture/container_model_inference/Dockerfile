# Use the watchdog as a base layer
FROM --platform=${TARGETPLATFORM:-linux/amd64} ghcr.io/openfaas/of-watchdog:0.9.15 as watchdog

# Use Debian-based Node.js image instead of Alpine
#FROM --platform=${TARGETPLATFORM:-linux/amd64} node:18-slim as ship
FROM rabbi29/base-container-image  as ship
# Set build arguments
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Copy watchdog
COPY --from=watchdog /fwatchdog /usr/bin/fwatchdog
RUN chmod +x /usr/bin/fwatchdog

# Install necessary packages on Debian
#RUN apt-get update && apt-get install -y --no-install-recommends \
     #build-essential  g++ python3-dev\
#     python3 python3-pip  python3-dev libjsoncpp-dev  \
#    && rm -rf /var/lib/apt/lists/*

# Create user and group
#ENV OPENSSL_DIR=/usr/lib/ssl
#RUN pip3 install torch==2.4.1+cpu --index-url https://download.pytorch.org/whl/cpu --break-system-packages
#RUN pip3 install pybind11 tokenizers  onnxruntime --break-system-packages
#RUN pip3 install 



RUN groupadd  app && useradd  -g app app && \
mkdir -p /home/app && \
chown -R app:app /home/app && \
chmod 755 /home/app

# Secure the environment and set permissions for root-only modifications
RUN echo 'export LD_PRELOAD=/usr/src/app/libmessage.so' > /etc/profile.d/preload.sh && \
    chmod 644 /etc/profile.d/preload.sh && \
    chown root:root /etc/profile.d/preload.sh

# Set working directory
#WORKDIR /usr/src/app

# Copy source files and Makefile into the container
#COPY common_entities/ ./

#RUN find / -name "json.h"

#ENV PYTHONPATH="/usr/src/app"

# Build the shared library
#RUN make all

# Ensure the library is owned by root and has correct permissions
#RUN chown root:root libmessage.so && \
#    chmod 755 libmessage.so

# Set environment variables for npm and permissions
ENV NPM_CONFIG_LOGLEVEL warn
RUN chmod 777 /tmp


# Switch to non-root user
USER app

# Set up function directory for app user
RUN mkdir -p /home/app/function

# Entrypoint setup
WORKDIR /home/app

COPY --chown=app:app package.json ./





# Install Node.js dependencies
RUN npm install

# Copy main function handler
COPY --chown=app:app index.js ./

# Set up function's dependencies separately for caching
WORKDIR /home/app/function
COPY --chown=app:app function/*.json ./
RUN npm install

# Copy function files and folders
COPY --chown=app:app function/ ./

# Set permissions and working directory
WORKDIR /home/app/

# Environment variables for fwatchdog
ENV cgi_headers="true"
ENV fprocess="node index.js"
ENV mode="http"
ENV upstream_url="http://127.0.0.1:3000"
ENV URL_GETPRICE="http://cci-narita.uncc.edu:29291/function/product-purchase-get-price"
ENV URL_AUTHORIZECC="http://cci-narita.uncc.edu:29291/function/product-purchase-authorize-cc"
ENV URL_PUBLISH="http://cci-narita.uncc.edu:29291/function/product-purchase-publish"

ENV exec_timeout="10s"
ENV write_timeout="15s"
ENV read_timeout="15s"
ENV prefix_logs="false"


RUN mkdir -p /tmp/strace_logs


ENV LD_PRELOAD=/usr/src/app/libmessage.so

# Healthcheck
HEALTHCHECK --interval=6s CMD [ -e /tmp/.lock ] || exit 1



# Set the command to start fwatchdog
CMD ["fwatchdog"]

#CMD ["strace", "-f", "-ff", "-o", "/tmp/strace_logs/trace_log", "fwatchdog"]
