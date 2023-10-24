FROM quay.io/fedora/fedora-minimal

COPY . /src/chromatic_shear_bias

WORKDIR /opt

RUN \
	curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" &&\
	/bin/bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge

ENV PATH=/opt/miniforge/bin:$PATH
ENV CONDA_PREFIX=/opt/miniforge

RUN \
	conda config --set solver libmamba &&\
	# conda env create --file=/src/chromatic_shear_bias/environment.yaml &&\
	conda install --yes --file=/src/chromatic_shear_bias/requirements.txt &&\
	conda clean --yes --all

RUN \
	pip install /src/chromatic_shear_bias

WORKDIR /home

