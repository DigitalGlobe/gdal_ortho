FROM pedros007/debian-gdal:2.3.0

COPY ./ /dg/staging/src/gdal_ortho/

RUN \
# Create AWS config file
    mkdir -p /root/.aws && \
    echo "[default]" >> /root/.aws/config && \
    echo "region = us-east-1" >> /root/.aws/config && \
# Install packages
    apt-get update && \
    apt-get -y install \
      ca-certificates \
      curl \
      libpq5 && \
    apt-get clean && \
# Install gdal_ortho
    pip install /dg/staging/src/gdal_ortho && \
# Remove temporary staging directory
    rm -rf /dg/staging && \
# Remove unneeded packages
    apt-get -y autoremove && \
# Clear pip cache
    rm -rf /root/.cache/pip && \
# Fix certificate path (some applications look in this alternate location)
    mkdir -p /etc/pki/tls/certs && \
    ln -s /etc/ssl/certs/ca-certificates.crt /etc/pki/tls/certs/ca-bundle.crt

