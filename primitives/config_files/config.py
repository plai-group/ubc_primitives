import os
from d3m import utils

try:
    import d3m.__init__ as d3m_info
    D3M_API_VERSION = d3m_info.__version__
except Exception:
    D3M_API_VERSION = '2019.4.4'

VERSION = "0.1.0"
TAG_NAME = "{git_commit}".format(git_commit=utils.current_git_commit(os.path.dirname(__file__)), )

REPOSITORY = "https://github.com/tonyjo/ubc_primitives.git"
PACAKGE_NAME = "ubc_primitives"

D3M_PERFORMER_TEAM = 'UBC'
D3M_CONTACT = "mailto:tonyjos@ubc.cs.ca"

if TAG_NAME:
    PACKAGE_URI = "git+" + REPOSITORY + "@" + TAG_NAME
else:
    PACKAGE_URI = "git+" + REPOSITORY

PACKAGE_URI = PACKAGE_URI + "#egg=" + PACAKGE_NAME


INSTALLATION_TYPE = 'GIT'
if INSTALLATION_TYPE == 'PYPI':
    INSTALLATION = {
        "type" : "PIP",
        "package": PACAKGE_NAME,
        "version": VERSION
    }
else:
    # INSTALLATION_TYPE == 'GIT'
    INSTALLATION = {
        "type" : "PIP",
        "package_uri": PACKAGE_URI,
    }
