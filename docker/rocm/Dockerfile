# Use ROCm base image
FROM rocm/dev-ubuntu-22.04:6.3-complete

# some graphics stuff for plotting
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglu1-mesa \
    libxrender1 \
    libxext6 \
    libsm6 \
    libxt6 \
    libx11-6 \
    libglx0 \
    libxcb1 \
    libfontconfig1

# get python stuff
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev python3-venv

COPY requirements.txt .
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt
# RUN pip install --upgrade pip && pip install -r requirements.txt

# Activate it for all future RUN, CMD, ENTRYPOINT
ENV PATH="/venv/bin:$PATH"

# Default to bash
CMD ["/bin/bash"]
