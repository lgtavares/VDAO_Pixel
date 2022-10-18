import setuptools
# python setup.py develop  # this places your "foo" package in the environment

setuptools.setup(
    name="VDAO_Pixel",
    version="1.0.0",
    author="Luiz Gustavo Tavares",
    author_email="luiz.tavares@smt.ufrj.br",
    description="Abandoned object detection in a cluttered environment",
    packages=["src"],
)
