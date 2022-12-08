from __future__ import annotations

import datetime
import os
import sys

now = datetime.datetime.now()

sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.graphviz',
]

# autodoc_mock_imports = ['bs4', 'requests']
add_module_names = False
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented_params'
autodoc_typehints_format = 'short'
autodoc_preserve_defaults = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# General information about the project.
project = 'femto'
copyright = str(now.year) + ', Riccardo Albiero'
author = 'Riccardo Albiero'

# version = femto.__version__
# release = femto.__version__

language = 'en'

pygments_style = 'sphinx'
todo_include_todos = False

# HTML
html_theme = 'sphinx_rtd_theme'
html_sidebars = {
    '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'],
}
html_show_sourcelink = False
htmlhelp_basename = 'femtodoc'
suppress_warnings = ['ref.python']
