FROM python:3.13.9-slim

# Avoid interactive apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Python quality-of-life settings
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /workspace

# Install bash and useful terminal tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    bash-completion \
    ca-certificates \
    coreutils \
    curl \
    git \
    grep \
    less \
    nano \
    procps \
    sed \
    unzip \
    zip \
    && rm -rf /var/lib/apt/lists/*

# Install pytest and the pinned project dependencies directly so this setup
# does not rely on uv.
RUN pip install --no-cache-dir --root-user-action=ignore -U pip pytest \
    "galois==0.4.4" \
    "numpy==2.1.3" \
    "numba==0.61.0" \
    "llvmlite==0.44.0"

# Make bash colorful for interactive use
RUN printf '\n\
# Color prompt\n\
export PS1="\\[\\e[1;32m\\]\\u@\\h\\[\\e[0m\\]:\\[\\e[1;34m\\]\\w\\[\\e[0m\\]\\$ "\n\
export TERM=xterm-256color\n\
alias ls="ls --color=auto"\n\
alias ll="ls -alF --color=auto"\n\
alias la="ls -A --color=auto"\n\
alias l="ls -CF --color=auto"\n' >> /etc/bash.bashrc

# Copy entrypoint
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
