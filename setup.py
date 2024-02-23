from distutils.core import setup

setup(
    name='alignment_tools',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.0.1',
    packages=['alignment_tools','alignment_tools.tests'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='alignment tools',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "pyqt5",
        "git+https://github.com/ElTinmar/qt_widgets.git@main"
    ]
)