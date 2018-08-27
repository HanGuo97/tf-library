# Using Kubernetes and NVIDIA Docker on DGX server

# NVIDIA Docker
### Using pre-built NVIDIA Docker Images
1. Go to [Nvidia Cloud](https://ngc.nvidia.com)
2. Generate API Key
3. Select preferred Docker (e.g. `nvcr.io/nvidia/tensorflow:18.07-py3`)

### Build customized Docker
1. Create a Dockerfile ([Best Practices](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/), Sample Dockerfile from [TF-Models](https://github.com/tensorflow/models/blob/master/official/Dockerfile.gpu), [OpenAI-Gym](https://github.com/openai/gym/blob/master/Dockerfile), [NVIDIA](https://gitlab.com/nvidia/samples/blob/master/cuda/ubuntu16.04/cuda-samples/Dockerfile), and [my own](https://github.com/HanGuo97/TF-RLLibs/blob/0.7/Dockerfile))
2. Build the docker `docker build -t DOCKER_TAG -f DOCKERFILE DIRECTORY/TO/CONTEXT`
3. Test the docker by running it in an interactive mode `docker run -ti --rm --runtime=nvidia DOCKER_TAG`, where `--runtime=nvidia` is required if the docker uses GPUs
4. Push the Docker into registries, e.g. [Docker Hub](https://docs.docker.com/get-started/part2/#share-your-image)
5. More can be found in [Docker Tutorial](https://docs.docker.com/get-started/)
6. GPU-enabled Docker can be found in this [blog post](https://devblogs.nvidia.com/gpu-containers-runtime/)

# Kubernetes

### Configuration
Use this [link](https://tarheellinux.unc.edu/?page_id=1024) to get user token. Currently, this token needs to be refreshed daily.

### Kubernetes Basics
```bash
kubectl get RESOURCE_TYPE [NAME] # List resource types, e.g. kubectl get pods, nodes. If NAME is specified, will list the resource type corresponding to the NAME, otherwise list all resource types
kubectl describe RESOURCE_TYPE [NAME] # Detailed report of resource types, e.g. kubectl describe node. If NAME is specified, will list the resource type corresponding to the NAME
kubectl delete -f CONFIG.yaml # Delete the object described in CONFIG.yaml
kubectl create -f CONFIG.yaml # Create the object described in CONFIG.yaml
```

### Configuration
```yaml
apiVersion: batch/v1
# Resource Type: Job, Deployment, Service, etc. Here we use Job
# https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/
kind: Job
metadata:
  # The name of the job
  name: NAME_OF_THE_JOB
spec:
  # https://kubernetes.io/docs/concepts/workloads/controllers/jobs-run-to-completion/#pod-backoff-failure-policy
  backoffLimit: 5
  template:
    spec:
      restartPolicy: Never # or OnFailure
      # Pull from Private Registry
      # https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/

      # to pull from a private repo, e.g. Docker Hub
      # https://stackoverflow.com/a/36974280

      # to pull from nvidia
      # kubectl create secret docker-registry SECRET_NAME \
      #     --docker-server=nvcr.io
      #     --docker-username="\$oauthtoken"
      #     --docker-password=PASSWORD
      #     --docker-email=EMAIL
      imagePullSecrets:
      - name: SECRET_NAME

      # Set the security context for a Pod
      # https://kubernetes.io/docs/tasks/configure-pod-container/security-context/#set-the-security-context-for-a-pod
      securityContext:
        runAsUser: USER_ID
        fsGroup: GROUP_ID

      # Volume could be: emptyPath, hostPath, etc
      # hostPath mounts host directories into pod
      # https://kubernetes.io/docs/concepts/storage/volumes/#hostpath
      volumes:
      - name: VOLUME_NAME
        hostPath:
          # directory location on host
          path: HOST_PATH
          # following field is optional
          # type: Directory

      containers:
      - name: CONTAINER_NAME
        # e.g. nvcr.io/nvidia/tensorflow:18.07-py3
        image: REPO_NAME/IMAGE_NAME:IMAGE_TAG
        imagePullPolicy: Always
        resources:
          # https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container/
          # You can specify GPU limits without specifying requests
          # because Kubernetes will use the limit as the request value by default.
          requests:
            cpu: CPU_REQUEST
            memory: MEMORY_REQUEST
          limits:
            cpu: CPU_REQUEST_LIMIT
            memory: MEMORY_REQUEST_LIMIT
            nvidia.com/gpu: GPU_REQUEST

        volumeMounts:
        # name must match the volume name defined in volumes
        - name: VOLUME_NAME
          # mount path within the container
          mountPath: MOUNT_PATH

        # https://kubernetes.io/docs/tasks/inject-data-application/define-command-argument-container/
        command: ["/bin/bash"]
        args: ["-c", "SHELL_COMMAND"]
```