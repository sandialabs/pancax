docker run -it -d --network=host --device=/dev/kfd --device=/dev/dri --ipc=host --shm-size 64G \
--group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -v $(pwd):/home/temp_user/pancax \
--name rocm_jax rocm/jax-community:rocm6.2.3-jax0.4.33-py3.12.6 /bin/bash

docker attach rocm_jax

