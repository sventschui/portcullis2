# Static hosting for the invite landing page and the universal-link association file.
# Runs unprivileged on 8080 behind a TLS-terminating reverse proxy (Traefik).
FROM nginxinc/nginx-unprivileged:1.28-alpine

# Drop everything the image ships that we do not serve.
USER root
RUN rm -rf /etc/nginx/conf.d /etc/nginx/nginx.conf /usr/share/nginx/html/* \
    && apk upgrade --no-cache
USER 101

COPY --chown=0:0 nginx.conf /etc/nginx/nginx.conf
COPY --chown=0:0 security-headers.conf /etc/nginx/security-headers.conf
COPY --chown=0:0 invite.html invite.css /srv/www/
COPY --chown=0:0 .well-known/apple-app-site-association /srv/www/.well-known/apple-app-site-association

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1
CMD ["nginx", "-g", "daemon off;"]
