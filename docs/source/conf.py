# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))
import sphinx.errors

sphinx.application.ExtensionError = sphinx.errors.ExtensionError

# -- Project information -----------------------------------------------------

project = "CompilerGym"
copyright = "Facebook AI Research"
author = "Facebook AI Research"

# Read the version from the //:VERSION file.
with open("../../VERSION") as f:
    version = f.read().strip()
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
    "sphinx.ext.autosectionlabel",
    "sphinxemoji.sphinxemoji",
]

autosectionlabel_prefix_document = True

templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "analytics_id": "G-T95G5EVYXM",
    "collapse_navigation": True,
    "display_version": True,
}

html_css_files = [
    "css/custom.css",
]

html_static_path = ["_static"]
