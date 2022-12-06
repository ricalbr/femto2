import sys
import os
import datetime
import femto

now = datetime.datetime.now()

sys.path.insert(0, os.path.abspath('../src'))

extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx.ext.doctest',
    'sphinx.ext.coverage',
    'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    'sphinx.ext.graphviz',
]

autodoc_mock_imports = ['bs4', 'requests']
add_module_names = False
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'autodoc_member_order': 'bysource',
    'autodoc_typehints': None,
    'autodoc_preserve_defaults': True,
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# General information about the project.
project = u'femto'
copyright = str(now.year) + u', Riccardo Albiero'
author = u'Riccardo Albiero'

# version = femto.__version__
# release = femto.__version__

language = None

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
