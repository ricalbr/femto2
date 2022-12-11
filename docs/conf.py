from __future__ import annotations

import datetime
import os
import shutil
import sys

now = datetime.datetime.now()

if os.path.exists('api'):
    shutil.rmtree('api')
os.system('sphinx-apidoc -feTo api ./../src/femto/  ./../src/femto/utils')

sys.path.insert(0, os.path.abspath('../src/'))

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

add_module_names = False
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'exclude-members': 'main',
    'undoc-members': False,
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
html_theme = 'furo'
html_sidebars = {
    '**': [
        'sidebar/scroll-start.html',
        'sidebar/brand.html',
        'sidebar/search.html',
        'sidebar/navigation.html',
        'sidebar/scroll-end.html',
    ]
}
html_show_sourcelink = False
htmlhelp_basename = 'femtodoc'
suppress_warnings = ['ref.python']
