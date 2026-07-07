
VERSION=0.1.3
DEVTEMPLATE_VERSION=0.6.18
PROJECT_NAME=ratecraft
PACKAGE_DIR=ratecraft
PROJECT_CONDA_PACKAGES=conda-packages.txt

# Conda environment paths for development
CONDA_PREFIX ?= /home/vscode/miniforge3

# Shared targets from dev-common (- prefix allows make to work before submodule init)
-include common/make/version.mk
-include common/make/utils.mk
-include common/make/python.mk
-include common/make/devcontainer.mk
-include stack-common/make/deploy.mk

# Project-specific targets below
