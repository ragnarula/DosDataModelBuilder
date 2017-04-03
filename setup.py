try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Model builder for denial of service data',
    'author': 'Raghav Narula',
    'author_email': 'dosmodelbuilder@raghavnarula.co.uk',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'DoS Data Model Builder',
    'entry_points': {
        'console_scripts': [
            'dos_model_builder = dos_model_builder.__main__:main'
        ]
    }
}

setup(**config)
