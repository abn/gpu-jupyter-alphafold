ARG CUDA=11.6

FROM docker.io/cschranz/gpu-jupyter:v1.4_cuda-${CUDA}_ubuntu-20.04

USER root

ARG ALPHAFOLD_COMMIT=86a0b8ec7a39698a7c2974420c4696ea4cb5743a
ARG HH_SUITE=3.3.0
ARG OPENMM=7.7.0
ARG ALPHAFOLD_PARAM_SOURCE_URL="https://storage.googleapis.com/alphafold/alphafold_params_colab_2022-03-02.tar"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        hmmer \
        kalign \
    && rm -rf /var/lib/apt/lists/*

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build \
    && pushd /tmp/hh-suite/build \
    && cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && popd \
    && rm -rf /tmp/hh-suite

# Install conda packages.
RUN conda install -y -c conda-forge \
      openmm=${OPENMM} \
      cudatoolkit \
      pdbfixer

RUN mkdir /opt/alphafold
RUN curl -sS -L \
    https://api.github.com/repos/deepmind/alphafold/tarball/${ALPHAFOLD_COMMIT} \
    | tar -xz --strip-components=1 -C /opt/alphafold

RUN wget -q -P /opt/alphafold/alphafold/common/ \
  https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

RUN mkdir -p /opt/alphafold/data/params
RUN curl -sS -L ${ALPHAFOLD_PARAM_SOURCE_URL} \
    | tar -xz -C /opt/alphafold/data/params

# Configure pip
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install additional packages
RUN pip --disable-pip-version-check install py3dmol

# Install jax
RUN pip --disable-pip-version-check install --upgrade \
      jax==0.3.16 \
      jaxlib==0.3.15+cuda11.cudnn82 \
      -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install alphafold (editable)
RUN pip install -e /opt/alphafold

# Make sure all necessary environment variables are set
ENV TF_FORCE_UNIFIED_MEMORY=1
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=2.0

RUN fix-permissions $CONDA_DIR
RUN fix-permissions /home/$NB_USER

USER $NB_UID
