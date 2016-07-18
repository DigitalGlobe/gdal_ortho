import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor

import click
from osgeo import gdal, osr

class CommandError(Exception): pass
class MetadataError(Exception): pass

# Constants
DEM_LAT_MARGIN_DEG = 1.0
DEM_LON_MARGIN_DEG = 1.0
GEOID_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "data",
                          "geoid_egm96-5_shifted.tif")
BAND_ALIASES = {
    "P": "PAN",
    "Multi": "MS",
    "MS1": "MS",
    "MS2": "MS"
}

# Initialize logging (root level WARNING, app level INFO)
logging_format = "[%(asctime)s|%(levelname)s|%(name)s|%(lineno)d] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=logging_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("-t_srs",
              "--target-srs",
              type=str,
              required=True,
              help="Target SRS for gdalwarp. Specify 'UTM' to auto-determine an approximate EPSG code for a UTM zone.")
@click.option("-ps",
              "--pixel-size",
              type=float,
              required=True,
              help="Pixel resolution in meters.")
@click.option("-rd",
              "--rpc-dem",
              type=click.Path(exists=True),
              default=None,
              help="Path to DEM for RPC calculations. (If omitted, image average elevation is used.) NOTE: "
              "The DEM is assumed to be in a projection where pixels are measured in degrees.")
@click.option("-geoid/-nogeoid",
              "--apply-geoid/--no-apply-geoid",
              default=True,
              help="Add (or don't add) geoid height to the --rpc-dem input. Refer to the DEM's vertical "
              "datum, DEMs referenced to the geoid need to be corrected to measure from the ellipsoid. Default "
              "is to add the geoid height.")
@click.option("-rm",
              "--resampling-method",
              type=str,
              default="cubic",
              help="Resampling method for gdalwarp. Default is cubic.")
@click.option("-vrt/-novrt",
              "--create-vrts/--no-create-vrts",
              default=False,
              help="Create (or skip) creation of VRTs, grouping files of like bands. Default is to skip VRT creation.")
@click.option("-cm",
              "--gdal-cachemax",
              type=click.IntRange(1, 2**20),
              default=8192,
              help="Per-process memory size in MB for GDAL cache (default 8192)")
@click.option("-wm",
              "--warp-memsize",
              type=click.IntRange(1, 2**20),
              default=2048,
              help="Per-process memory size in MB for warping (default 2048)")
@click.option("-wt",
              "--warp-threads",
              type=int,
              default=8,
              help="Number of threads for gdalwarp. Default is 8.")
@click.option("-np",
              "--num-parallel",
              type=int,
              default=2,
              help="Number of images to orthorectify in parallel. Default is 2.")
@click.option("-t",
              "--tmpdir",
              type=click.Path(exists=True),
              default=None,
              help="Local path for temporary files. (Default is OS-specific)")
def gdal_ortho(input_dir,
               output_dir,
               target_srs,
               pixel_size,
               rpc_dem,
               apply_geoid,
               resampling_method,
               create_vrts,
               gdal_cachemax,
               warp_memsize,
               warp_threads,
               num_parallel,
               tmpdir):
    """Wrapper for orthorectification using GDAL utilities.

    This script assumes that the necessary utilities are accessible
    from the execution environment.

    """

    # Fix paths
    input_dir = os.path.realpath(input_dir)
    output_dir = os.path.realpath(output_dir)
    if rpc_dem is not None:
        rpc_dem = os.path.realpath(rpc_dem)

    # Walk the input directory looking for TIFs to orthorectify. Also
    # store metadata extracted from each TIF's IMD file. While
    # gathering image info, store the best (minimum) GSD. This will be
    # used as a reference to scale the pixel size of images with a GSD
    # larger than the minimum.
    input_files = {}
    min_gsd = 1e9
    for (path, dirs, files) in os.walk(input_dir):
        for f in files:
            (basename, ext) = os.path.splitext(f)
            if ext.lower() == ".tif":
                # Check for matching IMD
                tif_path = os.path.join(path, f)
                imd_path = os.path.join(path, basename + ".IMD")
                if not os.path.isfile(imd_path):
                    raise MetadataError("No IMD found for %s" % tif_path)
                info = __get_imd_info(imd_path)
                input_files[tif_path] = info

                # Save best GSD
                min_gsd = min(min_gsd, info.avg_gsd)
    logger.info("Found %d images to orthorectify, best GSD is %.10f" % \
                (len(input_files), min_gsd))

    # Create a pool of worker threads. Each worker thread will call
    # out to GDAL utilities to do actual work.
    worker_pool = ThreadPoolExecutorWithCallback(max_workers=num_parallel)

    # Create a working directory for temporary data
    temp_dir = tempfile.mkdtemp(dir=tmpdir)
    output_files_by_band = defaultdict(list)
    try:
        # Loop over inputs and submit jobs to worker pool
        for (input_file, info) in input_files.iteritems():
            output_file = os.path.join(output_dir,
                                       os.path.relpath(input_file, input_dir))
            worker_pool.submit(worker_thread,
                               input_file,
                               info,
                               min_gsd,
                               target_srs,
                               pixel_size,
                               rpc_dem,
                               apply_geoid,
                               resampling_method,
                               gdal_cachemax,
                               warp_memsize,
                               warp_threads,
                               output_file,
                               temp_dir)

            # Add the output file to the finished list
            output_files_by_band[info.band_id].append(output_file)

        # Wait for all workers to finish
        canceled = False
        try:
            while worker_pool.is_running():
                time.sleep(1)
        except KeyboardInterrupt:
            canceled = True
            logger.warn("Received interrupt, canceling pending jobs...")
            num_canceled = worker_pool.cancel_all()
            logger.warn("Canceled %d pending jobs" % num_canceled)
            time.sleep(1)

        # Wait for workers to finish
        worker_pool.shutdown()

        if create_vrts and not canceled:
            # Create a VRT for each band
            for (band_id, output_files) in output_files_by_band.iteritems():
                # Check for a band alias to make a friendlier VRT name
                ### TODO: Maybe use the individual part names as a template?
                vrt_name = "ortho_%s.vrt" % band_id
                for (orig_id, alias_id) in BAND_ALIASES.iteritems():
                    if orig_id == band_id:
                        vrt_name = "ortho_%s.vrt" % alias_id
                        break

                # Get relative paths to files from the output directory
                relpaths = [os.path.relpath(f, output_dir) for f in output_files]

                # Create VRT (paths are relative to output_dir)
                logger.info("Creating band '%s' VRT %s" % (band_id, vrt_name))
                __run_cmd(["gdalbuildvrt",
                           "-srcnodata",
                           "0",
                           vrt_name] + relpaths,
                          fail_msg="Failed to band %s VRT %s" % (band_id, vrt_name),
                          cwd=output_dir)

    finally:
        # Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)

def worker_thread(input_file,
                  info,
                  min_gsd,
                  target_srs,
                  pixel_size,
                  rpc_dem,
                  apply_geoid,
                  resampling_method,
                  gdal_cachemax,
                  warp_memsize,
                  warp_threads,
                  output_file,
                  temp_dir):
    """Orthorectifies a single image using GDAL utilties.

    Args:
        input_file: Path to input 1B TIF.
        info: Tuple generated from __get_imd_info for the 1B TIF.
        min_gsd: Minimum (best) ground sample distance for all TIFs in
            the input directory, used to scale larger (worse) GSDs.
        target_srs: Spatial reference system to warp into.
        pixel_size: Requested output pixel size in meters.
        rpc_dem: Path to DEM to use for warping.
        apply_geoid: True to add geoid height to DEM, false to
            skip. Necessary for DEMs that are measured from
            geoid. (Most are.)
        resampling_method: Resampling method to use for warping.
        gdal_cachemax: Cache size to use for GDAL utilities.
        warp_memsize: Extra cache size for warping.
        warp_threads: Number of threads to use for warping.
        output_file: Path to output orthorectified file.
        temp_dir: Path to scratch directory for intermediate files.

    """

    logger.info("Processing %s" % input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Find the average location of the image and calculate the UTM
    # zone EPSG code. There are 60 zones around the globe. North zones
    # are EPSG:32601-EPSG:32660, south zones are
    # EPSG:32701-EPSG:32760.
    avg_lat = (info.min_lat + info.max_lat) / 2.0
    avg_lon = (info.min_lon + info.max_lon) / 2.0
    utm_epsg_code = int(((avg_lon + 180.0) / 6.0) + 1) + 32600 # 326xx
    if avg_lat < 0:
        utm_epsg_code += 100 # 327xx

    # Handle the special "UTM" target SRS
    if target_srs.lower() == "utm":
        this_target_srs = "EPSG:%d" % utm_epsg_code
    else:
        this_target_srs = target_srs

    # Check the unit of the target SRS and convert the pixel size if
    # necessary
    try:
        srs = osr.SpatialReference()
        srs.SetFromUserInput(str(this_target_srs))
        unit = srs.GetAttrValue("UNIT")
    finally:
        srs = None
    if unit.startswith("meter") or unit.startswith("metre"):
        # SRS unit is meters, no conversion necessary
        pixel_size_srs = pixel_size
    elif unit.startswith("degree"):
        # Convert to (approximate) degrees. 
        pixel_size_srs = (pixel_size * 360.0) / (osr.SRS_WGS84_SEMIMAJOR * 2 * math.pi)
    else:
        raise MetadataError("Unsupported target SRS %s unit type %s" % \
                            (this_target_srs, unit))

    # Scale the pixel size according to this image's GSD
    # compared to the maximum GSD of all the images.
    scale_ratio = info.avg_gsd / min_gsd
    this_pixel_size = pixel_size * scale_ratio
    this_pixel_size_srs = pixel_size_srs * scale_ratio

    # Check DEM input
    if rpc_dem is not None:
        # Subset DEM for this input including some
        # margin. NOTE: The DEM is assumed to be in a
        # projection where pixels are measured in degrees.
        min_lat = info.min_lat - DEM_LAT_MARGIN_DEG
        min_lon = info.min_lon - DEM_LON_MARGIN_DEG
        max_lat = info.max_lat + DEM_LAT_MARGIN_DEG
        max_lon = info.max_lon + DEM_LON_MARGIN_DEG
        dem_chip = os.path.join(temp_dir, base_name + "_DEM.tif")
        logger.info("Subsetting DEM, (lat, lon) = (%.10f, %.10f) - (%.10f, %.10f)" % \
                    (min_lat, min_lon, max_lat, max_lon))
        __run_cmd(["gdal_translate",
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
                  cwd=temp_dir)

        # Get the DEM's pixel resolution
        dem_pixel_size = None
        try:
            ds = gdal.Open(dem_chip)
            dem_pixel_size = ds.GetGeoTransform()[1]
        finally:
            ds = None
        if dem_pixel_size is None:
            raise MetadataError("Failed to get DEM chip %s pixel size" % dem_chip)
        logger.info("DEM pixel size is %.10f" % dem_pixel_size)

        # Check whether the DEM needs to be adjusted to height
        # above ellipsoid
        if apply_geoid:
            # Subset geoid to match the DEM chip
            geoid_chip = os.path.join(temp_dir, base_name + "_GEOID.tif")
            logger.info("Subsetting geoid, (lat, lon) = (%.10f, %.10f) - (%.10f, %.10f)" % \
                        (min_lat, min_lon, max_lat, max_lon))
            __run_cmd(["gdalwarp",
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
                      cwd=temp_dir)

            # Add the geoid to the DEM chip
            dem_plus_geoid_chip = os.path.join(temp_dir, base_name + "_DEM_PLUS_GEOID.tif")
            logger.info("Adding geoid to DEM")
            __run_cmd(["gdal_calc.py",
                       "-A",
                       dem_chip,
                       "-B",
                       geoid_chip,
                       "--calc",
                       "A+B",
                       "--outfile",
                       dem_plus_geoid_chip],
                      fail_msg="Failed to add geoid %s to DEM %s" % (geoid_chip, dem_chip),
                      cwd=temp_dir)
            dem_chip = dem_plus_geoid_chip

        # Orthorectify
        output_file_dir = os.path.dirname(output_file)
        logger.info("Orthorectifying to SRS %s, %.5f meter pixels" % \
                    (this_target_srs, this_pixel_size))
        if not os.path.isdir(output_file_dir):
            os.makedirs(output_file_dir)
        __run_cmd(["gdalwarp",
                   "--config",
                   "GDAL_CACHEMAX",
                   str(gdal_cachemax),
                   "-wm",
                   str(warp_memsize),
                   "-t_srs",
                   str(this_target_srs),
                   "-rpc",
                   "-tr",
                   str(this_pixel_size_srs),
                   str(this_pixel_size_srs),
                   "-r",
                   str(resampling_method),
                   "-multi",
                   "-wo",
                   "NUM_THREADS=%s" % warp_threads,
                   "-to",
                   "RPC_DEM=%s" % dem_chip,
                   "-to",
                   "RPC_DEMINTERPOLATION=bilinear",
                   "-co",
                   "TILED=YES",
                   input_file,
                   output_file],
                  fail_msg="Failed to orthorectify %s using DEM %s" % \
                  (input_file, dem_chip),
                  cwd=temp_dir)

    else:
        # Orthorectify using average height above ellipsoid
        output_file_dir = os.path.dirname(output_file)
        logger.info("Orthorectifying to SRS %s, %.5f meter pixels" % \
                    (this_target_srs, this_pixel_size))
        if not os.path.isdir(output_file_dir):
            os.makedirs(output_file_dir)
        __run_cmd(["gdalwarp",
                   "--config",
                   "GDAL_CACHEMAX",
                   str(gdal_cachemax),
                   "-wm",
                   str(warp_memsize),
                   "-t_srs",
                   str(this_target_srs),
                   "-rpc",
                   "-tr",
                   str(this_pixel_size_srs),
                   str(this_pixel_size_srs),
                   "-r",
                   str(resampling_method),
                   "-multi",
                   "-wo",
                   "NUM_THREADS=%s" % warp_threads,
                   "-to",
                   "RPC_HEIGHT=%s" % info.avg_hae,
                   "-co",
                   "TILED=YES",
                   input_file,
                   output_file],
                  fail_msg="Failed to orthorectify %s using average height %.10f" % \
                  (input_file, info.avg_hae),
                  cwd=temp_dir)

    # Copy the input file's IMD to the output location
    shutil.copy(info.imd_file,
                os.path.join(output_file_dir,
                             os.path.basename(info.imd_file)))

def __get_imd_info(imd_file):
    """Parses an IMD file for metadata.

    Args:
        imd_file: Path to IMD file to be parsed.

    Returns a namedtuple with the following fields (NOTE: each field
    is stored as a string):
        imd_file: Path to IMD file.
        band_id: Bands present in image.
        min_lat: Minimum latitude of image.
        min_lon: Minimum longitude of image.
        max_lat: Maximum latitude of image.
        max_lon: Maximum longitude of image.
        avg_hae: Average height above ellipsoid of image.
        avg_gsd: Average ground sample distance of image.

    """

    # Create return type
    InfoType = namedtuple("InfoType",
                          ["imd_file",
                           "band_id",
                           "min_lat",
                           "min_lon",
                           "max_lat",
                           "max_lon",
                           "avg_hae",
                           "avg_gsd"])

    # Read IMD contents
    with open(imd_file, "r") as f_obj:
        imd_str = f_obj.read()

    # Create a dict of all the values to read
    params = {
        "bandId": [],
        "ULLon": [],
        "ULLat": [],
        "ULHAE": [],
        "URLon": [],
        "URLat": [],
        "URHAE": [],
        "LRLon": [],
        "LRLat": [],
        "LRHAE": [],
        "LLLon": [],
        "LLLat": [],
        "LLHAE": [],
        "meanProductGSD": []
    }

    # Read each value
    for (param_name, param_val_list) in params.iteritems():
        for m_obj in re.finditer(r'%s\s*=\s*"?([^";]+)"?;' % param_name, imd_str):
            param_val_list.append(str(m_obj.group(1)))

    # Get the band ID
    if not params["bandId"]:
        raise MetadataError("Couldn't parse bandId in %s" % imd_file)
    band_id = params["bandId"][0]

    # Find float values
    latitudes = [float(val) for param in ["ULLat", "URLat", "LRLat", "LLLat"] for val in params[param]]
    longitudes = [float(val) for param in ["ULLon", "URLon", "LRLon", "LLLon"] for val in params[param]]
    heights = [float(val) for param in ["ULHAE", "URHAE", "LRHAE", "LLHAE"] for val in params[param]]
    gsds = [float(val) for val in params["meanProductGSD"]]

    return InfoType(imd_file=imd_file,
                    band_id=band_id,
                    min_lat=min(latitudes),
                    min_lon=min(longitudes),
                    max_lat=max(latitudes),
                    max_lon=max(longitudes),
                    avg_hae=sum(heights)/len(heights),
                    avg_gsd=sum(gsds)/len(gsds))

def __run_cmd(args, fail_msg="Command failed", cwd=None, env=None):
    # Get logger
    logger = logging.getLogger(__name__)

    # Convert args to strings
    string_args = [str(a) for a in args]
    logger.info("Running command: %s" % " ".join(['"%s"' % a for a in string_args]))

    # Spawn child process and wait for it to complete
    p_obj = subprocess.Popen(string_args,
                             cwd=cwd,
                             env=env)
    retval = p_obj.wait()
    if retval != 0:
        raise CommandError("%s, returned non-zero exit status %d" % \
                           (fail_msg, retval))

class ThreadPoolExecutorWithCallback(ThreadPoolExecutor):
    """ThreadPoolExecutor which automatically adds a callback to each future.

    The added callback is used to catch exceptions from worker
    processes. If an exception is caught, the callback sets a flag
    that can be checked externally.

    """

    def __init__(self, *args, **kwargs):
        super(ThreadPoolExecutorWithCallback, self).__init__(*args, **kwargs)
        self.futures = set()
        self.caught_exception = False

    def submit(self, fn, *args, **kwargs):
        future = super(ThreadPoolExecutorWithCallback, self).submit(fn, *args, **kwargs)
        future.add_done_callback(self.callback)
        self.futures.add(future)

    def callback(self, future):
        """Callback function for completed futures.

        Args:
            future: The future object to add the callback to.

        """
        self.futures.discard(future)

        # Don't check for exceptions if canceled
        if not future.cancelled():
            exc = future.exception()
            if exc is not None and not isinstance(exc, KeyboardInterrupt):
                self.caught_exception = True
                raise exc

    def cancel_all(self):
        """Attemps to cancel pending futures.

        Returns the number of futures that were canceled.

        """
        num_canceled = 0
        for future in list(self.futures):
            if future.cancel():
                num_canceled += 1
        return num_canceled

    def is_running(self):
        return len(self.futures) > 0

# Legacy entry point for calling script directly
if __name__ == "__main__":
    gdal_ortho()

