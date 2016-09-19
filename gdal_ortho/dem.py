import logging
import os
import sys

import boto3
import click
import fiona
import shapely.geometry
import shapely.ops
import tiletanic
import tiletanic.tilecover
import tiletanic.tileschemes
from botocore.exceptions import ClientError
from osgeo import gdal

from run_cmd import run_cmd

class InputError(Exception): pass

# Constants
GEOID_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "data",
                          "geoid_egm96-5_shifted.tif")
DEFAULT_DEM_MARGIN_DEG = 0.1
DGDEM_BUCKET = "dgdem"
DGDEM_CURRENT_KEY = "current.txt"
DGDEM_TILE_FORMAT = "DEM_%s.TIF"
DGDEM_GROUP_LEVEL = 3
DGDEM_TILE_LEVEL = 7

# Initialize logging (root level WARNING, app level INFO)
logging_format = "[%(asctime)s|%(levelname)s|%(name)s|%(lineno)d] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=logging_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def local_dem_chip(base_name,
                   dem_dir,
                   bbox,
                   rpc_dem,
                   apply_geoid,
                   gdal_cachemax,
                   warp_memsize,
                   warp_threads,
                   margin=DEFAULT_DEM_MARGIN_DEG):
    """Subsets a local DEM for a specific area.

    Args:
        base_name: Base name of generated files.
        dem_dir: Path to destination directory for DEM chip.
        bbox: Shapely geometry containing area to extract.
        rpc_dem: Path to local DEM.
        apply_geoid: True to add geoid height to DEM, false to
            skip. Necessary for DEMs that are measured from
            geoid. (Most are.)
        gdal_cachemax: Cache size to use for GDAL utilities.
        warp_memsize: Extra cache size for warping.
        warp_threads: Number of threads to use for warping.

    Returns the path to the DEM chip.

    """

    # Subset DEM for this input including some margin. NOTE: The DEM
    # is assumed to be in a projection where pixels are measured in
    # degrees.
    bounds = bbox.bounds
    min_lat = bounds[1] - margin
    min_lon = bounds[0] - margin
    max_lat = bounds[3] + margin
    max_lon = bounds[2] + margin
    dem_chip = os.path.join(dem_dir, base_name + "_DEM.tif")
    logger.info("Subsetting DEM, (lat, lon) = (%.10f, %.10f) - (%.10f, %.10f)" % \
                (min_lat, min_lon, max_lat, max_lon))
    run_cmd(["gdal_translate",
             "--config",
             "GDAL_CACHEMAX",
             str(gdal_cachemax),
             "-projwin",
             str(min_lon),
             str(max_lat),
             str(max_lon),
             str(min_lat),
             rpc_dem,
             dem_chip],
            fail_msg="Failed to subset DEM %s" % rpc_dem,
            cwd=dem_dir)

    # Get the DEM's pixel resolution
    dem_pixel_size = None
    try:
        ds = gdal.Open(dem_chip)
        dem_pixel_size = ds.GetGeoTransform()[1]
    finally:
        ds = None
    if dem_pixel_size is None:
        raise InputError("Failed to get DEM chip %s pixel size" % dem_chip)
    logger.info("DEM pixel size is %.10f" % dem_pixel_size)

    #  whether the DEM needs to be adjusted to height above ellipsoid
    if apply_geoid:
        # Subset geoid to match the DEM chip
        geoid_chip = os.path.join(dem_dir, base_name + "_GEOID.tif")
        logger.info("Subsetting geoid, (lat, lon) = (%.10f, %.10f) - (%.10f, %.10f)" % \
                    (min_lat, min_lon, max_lat, max_lon))
        run_cmd(["gdalwarp",
                 "--config",
                 "GDAL_CACHEMAX",
                 str(gdal_cachemax),
                 "-wm",
                 str(warp_memsize),
                 "-t_srs",
                 "EPSG:4326",
                 "-te",
                 str(min_lon),
                 str(min_lat),
                 str(max_lon),
                 str(max_lat),
                 "-tr",
                 str(dem_pixel_size),
                 str(dem_pixel_size),
                 "-r",
                 "bilinear",
                 GEOID_PATH,
                 geoid_chip],
                fail_msg="Failed to subset geoid %s" % GEOID_PATH,
                cwd=dem_dir)

        # Add the geoid to the DEM chip
        dem_plus_geoid_chip = os.path.join(dem_dir, base_name + "_DEM_PLUS_GEOID.tif")
        logger.info("Adding geoid to DEM")
        run_cmd(["gdal_calc.py",
                 "-A",
                 dem_chip,
                 "-B",
                 geoid_chip,
                 "--calc",
                 "A+B",
                 "--outfile",
                 dem_plus_geoid_chip],
                fail_msg="Failed to add geoid %s to DEM %s" % (geoid_chip, dem_chip),
                cwd=dem_dir)
        dem_chip = dem_plus_geoid_chip

    return dem_chip

def fetch_dgdem_tiles(aoi_geom, output_vrt, margin=DEFAULT_DEM_MARGIN_DEG):
    """Fetches DEM tiles from the GBDX s3://dgdem bucket.

    The DEM is stored in level 7 tiles. This function fetches the
    tiles required to cover an AOI and generates a VRT to bundle all
    the tiles together.

    The entire DEM is stored under a prefix with a date code. Below
    that, tiles are grouped into prefixes for each level 3
    block. Below that, each level 7 tile is named DEM_<quadkey>.tif.

    The prefix to use for the latest DEM version is stored in a text
    file at s3://dgdem/current.txt.

    Args:
        aoi_geom: Shapely geometry defining the AOI to cover.
        output_vrt: Filename of output VRT containing the DEM tiles
            that cover the AOI.
        margin: Margin in degrees to buffer around AOI.

    Returns True if the tiles were downloaded successfully, False if not.

    """

    # Get the quadkeys that cover the AOI
    scheme = tiletanic.tileschemes.DGTiling()
    tile_gen = tiletanic.tilecover.cover_geometry(scheme, aoi_geom, DGDEM_TILE_LEVEL)
    qks = [scheme.quadkey(tile) for tile in tile_gen]
    logger.info("Found %d quadkeys that cover the AOI: %s" % \
                (len(qks), ", ".join(qks)))

    # Download the pointer file that says what DEM prefix to use
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(DGDEM_BUCKET)
    pointer = bucket.Object(DGDEM_CURRENT_KEY)
    prefix = pointer.get()["Body"].read().strip()

    # Generate S3 keys to download
    def key_from_quadkey(qk):
        group = qk[0:DGDEM_GROUP_LEVEL]
        return os.path.join(prefix, group, DGDEM_TILE_FORMAT % qk)
    key_map = {qk:key_from_quadkey(qk) for qk in qks}

    # Verify that all keys are available
    tiles_missing = False
    for (qk, key) in key_map.iteritems():
        try:
            obj = bucket.Object(key)
            obj.load()
        except ClientError as exc:
            if "404" not in exc.message:
                raise exc
            else:
                logger.warn("Missing S3 object s3://%s/%s for quadkey %s" % \
                            (DGDEM_BUCKET, key, qk))
                tiles_missing = True
    if tiles_missing:
        return False

    # Download DEM tiles from S3
    out_dir = os.path.dirname(output_vrt)
    dem_files = []
    for (qk, key) in key_map.iteritems():
        dem_filename = os.path.join(out_dir, key)
        dem_filename_dir = os.path.dirname(dem_filename)
        if not os.path.isdir(dem_filename_dir):
            os.makedirs(dem_filename_dir)
        dem_files.append(dem_filename)
        logger.info("Downloading quadkey %s tile s3://%s/%s to %s" % \
                    (qk, DGDEM_BUCKET, key, dem_filename))
        bucket.download_file(key, dem_filename)

    # Generate VRT containing DEM tiles
    logger.info("Creating VRT " + output_vrt)
    run_cmd(["gdalbuildvrt",
             "-vrtnodata",
             "None",
             output_vrt] + dem_files,
            fail_msg="Failed to create DEM VRT")
    return True

@click.command()
@click.option("-aoi",
              "--aoi",
              required=True,
              type=click.Path(exists=True),
              help="OGR-compatible vector file containing AOI to cover.")
@click.option("-vrt",
              "--output-vrt",
              required=True,
              type=click.Path(exists=False),
              help="Output VRT containing the DEM tiles that cover the AOI.")
@click.option("-m",
              "--margin",
              type=float,
              default=DEFAULT_DEM_MARGIN_DEG,
              help="Margin in degrees to buffer around AOI. (Default %s deg)" % \
              DEFAULT_DEM_MARGIN_DEG)
def fetch_dgdem(aoi, output_vrt, margin):
    """Fetches DEM tiles from the GBDX s3://dgdem bucket.

    The DEM is stored in level 7 tiles. This function fetches the
    tiles required to cover an AOI and generates a VRT to bundle all
    the tiles together.

    The entire DEM is stored under a prefix with a date code. Below
    that, tiles are grouped into prefixes for each level 3
    block. Below that, each level 7 tile is named DEM_<quadkey>.tif.

    The prefix to use for the latest DEM version is stored in a text
    file at s3://dgdem/current.txt.

    """

    # Read records from the input AOI and union into a single shape
    with fiona.open(aoi, "r") as shp:
        recs = [rec for rec in shp]
    geoms = [shapely.geometry.shape(rec["geometry"]) for rec in recs]
    aoi_geom = shapely.ops.unary_union(geoms).buffer(margin)

    # Download the tiles
    fetch_dgdem_tiles(aoi_geom, os.path.realpath(output_vrt), margin)

