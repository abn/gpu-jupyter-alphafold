ARG CUDA=11.7.1
ARG UBUNTU=22.04

FROM docker.io/jupyter/base-notebook:ubuntu-${UBUNTU} AS base-notebook


FROM docker.io/nvidia/cuda:${CUDA}-cudnn8-devel-ubuntu${UBUNTU} AS alphafold-build

ARG ALPHAFOLD_COMMIT=86a0b8ec7a39698a7c2974420c4696ea4cb5743a
ARG ALPHAFOLD_PARAM_SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-03-02.tar"
ARG HH_SUITE=3.3.0

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

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

# Compile HHsuite from source.
RUN git clone --branch v${HH_SUITE} https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && cd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && cd - \
    && rm -rf /tmp/hh-suite

RUN mkdir /opt/alphafold \
    && curl -sS -L \
        https://api.github.com/repos/deepmind/alphafold/tarball/${ALPHAFOLD_COMMIT} \
        | tar -xz --strip-components=1 -C /opt/alphafold

RUN wget -q -P /opt/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN mkdir -p /opt/alphafold/data/params \
    && curl -sS -L ${ALPHAFOLD_PARAM_SOURCE_URL} \
        | tar -x --preserve-permissions --no-same-owner -C /opt/alphafold/data/params

ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"

COPY --from=base-notebook /opt/conda /opt/conda

RUN pip wheel jax==0.3.16 \
      jaxlib==0.3.15+cuda11.cudnn82 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

RUN cd /opt/alphafold \
    && python setup.py egg_info \
    && cat alphafold.egg-info/requires.txt \
        | grep -Ev "^(docker|jax|jaxlib)$" \
        | xargs -I% pip install %

RUN conda install -y -c conda-forge \
      openmm \
      cudatoolkit \
      pdbfixer \
      py3dmol

# Install alphafold (editable)
RUN pip install --no-deps -e /opt/alphafold

FROM docker.io/nvidia/cuda:${CUDA}-cudnn8-runtime-ubuntu${UBUNTU}

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

USER 0

# Fix: https://github.com/hadolint/hadolint/wiki/DL4006
# Fix: https://github.com/koalaman/shellcheck/wiki/SC3014
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Configure environment
ENV CONDA_DIR=/opt/conda \
    ALPHAFOLD_DIR=/opt/alphafold \
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

ENV PATH="${CONDA_DIR}/bin:${PATH}" \
    HOME="/home/${NB_USER}"

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

COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build /opt/conda /opt/conda
COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build /opt/hhsuite /opt/hhsuite
COPY --chown="${NB_UID}:${NB_GID}" --from=alphafold-build /opt/alphafold /opt/alphafold

RUN ln -s /opt/hhsuite/bin/* /usr/bin

RUN useradd -l -M -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" \
    && chmod g+w /etc/passwd \
    && fix-permissions "${HOME}" \
    && fix-permissions "${CONDA_DIR}" \
    && fix-permissions "${ALPHAFOLD_DIR}" \
    && fix-permissions /etc/jupyter

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
