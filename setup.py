from distutils.core import setup

setup(
    name='alignment_tools',
    python_requires='>=3.8',
    author='Martin Privat',
    version='0.1.0',
    packages=['alignment_tools'],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    description='alignment tools',
    long_description=open('README.md').read(),
    install_requires=[
        "numpy", 
        "PyQt5",
        "opencv-python",
        "antspyx",
        "qt_widgets @ git+https://github.com/ElTinmar/qt_widgets.git@main",
        "image_tools @ git+https://github.com/ElTinmar/image_tools.git@main"
    ]
)
