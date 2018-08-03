from setuptools import setup, find_packages

setup(name="gdal_ortho",
      version="0.1",
      packages=find_packages(exclude=["test*"]),
      package_data={"gdal_ortho": ["data/*"]},
      install_requires=[
          "awscli",
          "boto3",
          "Click",
          "fiona",
          "futures",
          "shapely",
          "tiletanic"
      ],
      entry_points='''
          [console_scripts]
          gdal_ortho=gdal_ortho.gdal_ortho:gdal_ortho
          aoi_to_srs=gdal_ortho.gdal_ortho:aoi_to_srs
          fetch_dem=gdal_ortho.dem:fetch_dem
      ''')

