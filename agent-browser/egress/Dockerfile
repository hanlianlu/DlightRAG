# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
#
# The Agent Browser's egress proxy (ADR 0032): the pool containers sit on internal
# networks only, and this is their one way out. squid.conf admits public destinations
# only.
FROM ubuntu:24.04
RUN apt-get update \
 && apt-get install -y --no-install-recommends squid \
 && rm -rf /var/lib/apt/lists/*
COPY squid.conf /etc/squid/squid.conf
USER proxy
EXPOSE 3128
CMD ["squid","-N","-d","1","-f","/etc/squid/squid.conf"]
