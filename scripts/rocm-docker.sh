docker run -it \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --ipc=host \
    --shm-size 64G \
    --group-add video --cap-add=SYS_PTRACE \
    --security-opt \
    seccomp=unconfined \
    -v $(pwd):/home/temp_user/pancax \
pancax /bin/bash
