ARG CUDA=12.1.1
ARG UBUNTU=22.04
ARG BASE=runtime
ARG LAB=4.0.6

FROM docker.io/jupyter/base-notebook:lab-${LAB} AS base-notebook

FROM nvcr.io/nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu${UBUNTU} AS alphafold-build

ARG CONDA_DIR=/opt/conda
ARG HHSUITE_DIR=/opt/hhsuite
ARG HHSUITE_VERSION=3.3.0
ARG ALPHAFOLD_GH_PROJECT=deepmind/alphafold
ARG ALPHAFOLD_DIR=/opt/alphafold
ARG ALPHAFOLD_COMMIT=a3941673e90b8d1d75c60b16a4b3707ebf7ba527
ARG PARALLELFOLD_GH_PROJECT=zuricho/parallelfold
ARG PARALLELFOLD_DIR=/opt/parallelfold
ARG PARALLELFOLD_COMMIT=02303ce7903a243e8c8cddfc7fce45b65f0e6713

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        build-essential \
        cmake \
        git \
        hmmer \
        kalign \
        tzdata \
        wget \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install hhsuite static binaries (lacks mpi support)
RUN mkdir -p ${HHSUITE_DIR} \
    && curl -sSL https://github.com/soedinglab/hh-suite/releases/download/v${HHSUITE_VERSION}/hhsuite-${HHSUITE_VERSION}-AVX2-Linux.tar.gz \
        | tar -xzv -C ${HHSUITE_DIR}

RUN mkdir ${ALPHAFOLD_DIR} \
    && curl -sS -L \
        https://api.github.com/repos/${ALPHAFOLD_GH_PROJECT}/tarball/${ALPHAFOLD_COMMIT} \
        | tar -xz --strip-components=1 -C ${ALPHAFOLD_DIR}

# install parallelfold
RUN mkdir ${PARALLELFOLD_DIR} \
    && curl -sS -L \
        https://api.github.com/repos/${PARALLELFOLD_GH_PROJECT}/tarball/${PARALLELFOLD_COMMIT} \
        | tar -xz --strip-components=1 -C ${PARALLELFOLD_DIR}


RUN wget -q -P ${ALPHAFOLD_DIR}/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt
RUN cp ${ALPHAFOLD_DIR}/alphafold/common/stereo_chemical_props.txt ${PARALLELFOLD_DIR}/alphafold/common/

ENV PATH="${CONDA_DIR}/bin:${HHSUITE_DIR}/bin:${HHSUITE_DIR}/scripts:${PATH}"

COPY --from=base-notebook ${CONDA_DIR} ${CONDA_DIR}

RUN pip install jax==0.4.14 \
      jaxlib==0.4.14+cuda${CUDA_VERSION%%.*}.cudnn$([[ ${CUDA_VERSION%%.*} = 12 ]] && echo 89 || echo 86) \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN pip install tensorflow

RUN cd ${ALPHAFOLD_DIR} \
    && python setup.py egg_info \
    && cat alphafold.egg-info/requires.txt \
        | grep -Ev "^(docker|jax|jaxlib|tensorflow-cpu)$" \
        | xargs -I% pip install %

RUN conda install -y -c conda-forge \
      openmm \
      cudatoolkit \
      pdbfixer \
      # additional useful packages
      py3dmol \
      matplotlib

RUN conda install -y -c bioconda \
      kalign2

RUN conda clean -afy \
    && find ${CONDA_DIR} -follow -type f -name '*.a' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete \
    && find ${CONDA_DIR} -follow -type f -name '*.js.map' -delete

# Install alphafold (editable)
RUN pip install --no-deps -e ${ALPHAFOLD_DIR}

# Out-of-tree patches (alphafold)
RUN sh -c 'find ${ALPHAFOLD_DIR} -type f -name "*.py" -exec sed -i s/simtk.openmm/openmm/ {} \;'

# Out-of-tree patches (parallelfold)
RUN sh -c 'find ${PARALLELFOLD_DIR} -type f -name "*.py" -exec sed -i s/simtk.openmm/openmm/ {} \;'
RUN sh -c 'find ${PARALLELFOLD_DIR} -type f -name "*.py" -exec sed -i s/collections.Iterable/collections.abc.Iterable/ {} \;'
RUN sh -c 'sed -i s/jax.tree_multimap/jax.tree_map/ ${PARALLELFOLD_DIR}/alphafold/model/mapping.py'


FROM nvcr.io/nvidia/cuda:${CUDA}-cudnn8-${BASE}-ubuntu${UBUNTU}

ARG CONDA_DIR=/opt/conda
ARG HHSUITE_DIR=/opt/hhsuite
ARG ALPHAFOLD_DIR=/opt/alphafold
ARG PARALLELFOLD_DIR=/opt/parallelfold

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER 0

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Configure environment
ENV CONDA_DIR=${CONDA_DIR} \
    ALPHAFOLD_DIR=${ALPHAFOLD_DIR} \
    PARALLELFOLD_DIR=${PARALLELFOLD_DIR} \
    SHELL=/bin/bash \
    NB_USER="${NB_USER}" \
    NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_FORCE_UNIFIED_MEMORY=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=2.0

ENV PATH="${CONDA_DIR}/bin:${HHSUITE_DIR}/bin:${HHSUITE_DIR}/scripts:${PATH}"
ENV HOME="/home/${NB_USER}"

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN apt-get update --yes \
    && apt-get upgrade --yes \
    && apt-get install --yes --no-install-recommends \
        # jupyter base
        bzip2 \
        ca-certificates \
        fonts-liberation \
        locales \
        pandoc \
        run-one \
        sudo \
        tini \
        wget \
        # alphafold
        hmmer \
        kalign \
        # extras
        curl \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen \
    && locale-gen

COPY --from=base-notebook /usr/local/bin /usr/local/bin
COPY --from=base-notebook /etc/jupyter /etc/jupyter
COPY --from=base-notebook /etc/pam.d/su /etc/pam.d/su
COPY --from=base-notebook /etc/sudoers /etc/sudoers
COPY --from=base-notebook /etc/skel/.bashrc /etc/skel/.bashrc

COPY --chown="${NB_UID}:${NB_GID}" --from=base-notebook /home/${NB_USER} /home/${NB_USER}

COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build ${CONDA_DIR} ${CONDA_DIR}
COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build ${HHSUITE_DIR} ${HHSUITE_DIR}
COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build ${ALPHAFOLD_DIR} ${ALPHAFOLD_DIR}
COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build ${PARALLELFOLD_DIR} ${PARALLELFOLD_DIR}

# install additional packages
RUN pip install ipyfilechooser

RUN useradd -l -M -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" \
    && chmod g+w /etc/passwd \
    && fix-permissions "${HOME}" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "${ALPHAFOLD_DIR}" \
    && fix-permissions "${PARALLELFOLD_DIR}" \
    && fix-permissions /etc/jupyter

# Add SETUID bit to the ldconfig binary so that non-root users can run it.
RUN chmod u+s /sbin/ldconfig{,.real}

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for
# details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.

# alphafold
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python ${ALPHAFOLD_DIR}/run_alphafold.py "$@"' > /usr/local/bin/alphafold \
  && chmod +x /usr/local/bin/alphafold

# parallelfold
RUN echo $'#!/bin/bash\n\
ldconfig\n\
PYTHONPATH=${PARALLELFOLD_DIR} python ${PARALLELFOLD_DIR}/run_alphafold.py "$@"' > /usr/local/bin/parallelfold \
  && chmod +x /usr/local/bin/parallelfold

USER ${NB_UID}
WORKDIR "${HOME}"

EXPOSE 8888

# Configure container startup
ENTRYPOINT ["tini", "-g", "--"]
CMD ["start-notebook.sh"]

# HEALTHCHECK documentation: https://docs.docker.com/engine/reference/builder/#healthcheck
# This healtcheck works well for `lab`, `notebook`, `nbclassic`, `server` and `retro` jupyter commands
# https://github.com/jupyter/docker-stacks/issues/915#issuecomment-1068528799
HEALTHCHECK  --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -O- --no-verbose --tries=1 --no-check-certificate \
    http${GEN_CERT:+s}://localhost:8888${JUPYTERHUB_SERVICE_PREFIX:-/}api || exit 1
