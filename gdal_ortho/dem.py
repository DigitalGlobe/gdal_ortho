import logging
import os
import re
import sys
from urlparse import urlparse

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

# Initialize logging (root level WARNING, app level INFO)
logging_format = "[%(asctime)s|%(levelname)s|%(name)s|%(lineno)d] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=logging_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def parse_s3_url(url):
    """Parses an S3 URL to extract the bucket and key.

    This function handles a few different S3 URL formats:
        http://s3.amazonaws.com/bucket/key
        http://bucket.s3.amazonaws.com/key
        s3://bucket/key

    Args:
        url: The S3 URL to parse.

    Returns a tuple containing (bucket, key). If the provided URL is
    not a valid S3 URL, an empty bucket and key are returned,
    i.e. ("", "").

    """

    # Parse URL
    parsed_url = urlparse(url)
    netloc = parsed_url.netloc
    path = parsed_url.path
    if path.startswith("/"):
        path = path[1:]

    # Initialize empty bucket/key and check the hostname
    bucket = ""
    key = ""
    if netloc == "s3.amazonaws.com":
        # Bucket is the first part of the path
        split_path = path.split("/")
        if len(split_path) >= 2:
            bucket = split_path[0]
            key = "/".join(split_path[1:])
    elif netloc:
        # Hostname is not empty, so bucket is in the netloc as a
        # "virtual host"
        m_obj = re.search(r"([^\.]+)", netloc)
        if m_obj is not None:
            bucket = m_obj.group(1)
            key = path

    return (bucket, key)

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

    # Check whether the DEM needs to be adjusted to height above ellipsoid
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

def download_tiles(dem_url, aoi_geom, output_vrt, margin=DEFAULT_DEM_MARGIN_DEG):
    """Fetches DEM tiles from S3.

    If the specified URL is a single object, it is assumed to be a
    text file containing the prefix of the DEM in the current
    bucket. This can be used like a symbolic link to point at the most
    recent version in the bucket. For example, the GBDX account
    contains a factory DEM at s3://dgdem/current.txt. Inside of
    current.txt is the actual prefix, "DEM_2016_08_12". So tiles are
    stored under s3://dgdem/DEM_2016_08_12.

    If the specified URL is not a single object (i.e. it is a prefix),
    DEM tiles are assumed to be stored within it. The tiles are
    assumed to be in the EPSG:4326 DG tiling scheme.

    Args:
        dem_url: S3 URL of DEM tiles.
        aoi_geom: Shapely geometry defining the AOI to cover.
        output_vrt: Filename of output VRT containing the DEM tiles
            that cover the AOI.
        margin: Margin in degrees to buffer around AOI.

    Returns True if the tiles were downloaded successfully, False if not.

    """

    # Check whether the DEM URL is a prefix
    s3 = boto3.resource("s3")
    (bucket_name, key) = parse_s3_url(dem_url)
    obj = s3.Object(bucket_name, key)
    try:
        obj.content_length
        is_prefix = False
        logger.info("%s is a pointer, downloading contents" % dem_url)
    except ClientError as exc:
        if "404" in str(exc):
            is_prefix = True
            logger.info("%s is a prefix" % dem_url)
        else:
            logger.error(str(exc))
            return False
    bucket = s3.Bucket(bucket_name)

    # If the URL is not a prefix, open it to find the actual prefix
    if not is_prefix:
        # Download file with prefix in it
        pointer = bucket.Object(key)
        prefix = pointer.get()["Body"].read().strip()
    else:
        # URL is already a prefix
        prefix = key

    # Get the tile keys
    logger.info("Searching for tiles under s3://%s/%s" % (bucket_name, prefix))
    dem_tiles = {}
    for obj in bucket.objects.filter(Prefix=prefix):
        mobj = re.search("([0-3]+)[^/]+$", obj.key)
        if mobj is not None:
            dem_tiles[mobj.group(1)] = obj.key
    if not dem_tiles:
        logger.error("No DEM tiles found in bucket %s under prefix %s" % \
                     (bucket_name, prefix))
        return False

    # Grab a key and determine the zoom level
    zoom_level = len(dem_tiles.keys()[0])
    logger.info("Found %d tiles, assuming zoom level %d" % (len(dem_tiles), zoom_level))

    # Get the quadkeys that cover the AOI
    scheme = tiletanic.tileschemes.DGTiling()
    tile_gen = tiletanic.tilecover.cover_geometry(scheme, aoi_geom, zoom_level)
    qks = [scheme.quadkey(tile) for tile in tile_gen]
    missing_qks = [qk for qk in qks if qk not in dem_tiles]
    if missing_qks:
        logger.error("%d quadkeys missing from DEM: %s" % \
                     (len(missing_qks), ", ".join(missing_qks)))
        return False
    logger.info("Found %d quadkeys that cover the AOI: %s" % \
                (len(qks), ", ".join(qks)))

    # Download DEM tiles from S3
    out_dir = os.path.dirname(output_vrt)
    dem_files = []
    for qk in qks:
        key = dem_tiles[qk]
        dem_filename = os.path.join(out_dir, key)
        dem_filename_dir = os.path.dirname(dem_filename)
        if not os.path.isdir(dem_filename_dir):
            os.makedirs(dem_filename_dir)
        dem_files.append(dem_filename)
        logger.info("Downloading quadkey %s tile s3://%s/%s to %s" % \
                    (qk, bucket_name, key, dem_filename))
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
@click.option("-url",
              "--dem-url",
              type=str,
              default="s3://dgdem/current.txt",
              help="S3 URL of DEM. (Default s3://dgdem/current.txt)")
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
def fetch_dem(dem_url, aoi, output_vrt, margin):
    """Fetches DEM tiles from S3.

    If the specified URL is a single object, it is assumed to be a
    text file containing the prefix of the DEM in the current
    bucket. This can be used like a symbolic link to point at the most
    recent version in the bucket. For example, the GBDX account
    contains a factory DEM at s3://dgdem/current.txt. Inside of
    current.txt is the actual prefix, "DEM_2016_08_12". So tiles are
    stored under s3://dgdem/DEM_2016_08_12.

    If the specified URL is not a single object (i.e. it is a prefix),
    DEM tiles are assumed to be stored within it. The tiles are
    assumed to be in the EPSG:4326 DG tiling scheme.

    """

    # Read records from the input AOI and union into a single shape
    with fiona.open(aoi, "r") as shp:
        recs = [rec for rec in shp]
    geoms = [shapely.geometry.shape(rec["geometry"]) for rec in recs]
    aoi_geom = shapely.ops.unary_union(geoms).buffer(margin)

    # Download the tiles
    download_tiles(dem_url, aoi_geom, os.path.realpath(output_vrt), margin)

