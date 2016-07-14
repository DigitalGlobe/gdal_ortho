from setuptools import setup, find_packages

setup(name="gdal_ortho",
      version="0.1",
      packages=find_packages(exclude=["test*"]),
      package_data={"gdal_ortho": ["data/*"]},
      install_requires=[
          "Click"
      ],
      entry_points='''
          [console_scripts]
          gdal_ortho=gdal_ortho.gdal_ortho:gdal_ortho
      ''')

