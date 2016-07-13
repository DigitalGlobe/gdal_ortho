import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict, namedtuple

import click
from osgeo import gdal, osr

class CommandError(Exception): pass
class MetadataError(Exception): pass

# Constants
DEM_LAT_MARGIN_DEG = 1.0
DEM_LON_MARGIN_DEG = 1.0
GDAL_CACHE_SIZE_MB = 8192
WARP_CACHE_SIZE_MB = 2048
GEOID_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "data",
                          "geoid_egm96-5_shifted.tif")
BAND_ALIASES = {
    "P": "PAN",
    "Multi": "MS",
    "MS1": "MS",
    "MS2": "MS"
}

@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
@click.option("-t_srs",
              "--target-srs",
              type=str,
              required=True,
              help="Target SRS for gdalwarp.")
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
              help="Apply (or don't apply) geoid to --rpc-dem input. (Refer to DEM's vertical datum, DEMs "
              "referenced to ellipsoid need to be corrected to geoid.) Default is to apply geoid.")
@click.option("-rm",
              "--resampling-method",
              type=str,
              default="cubic",
              help="Resampling method for gdalwarp. Default is cubic.")
@click.option("-vrt/-novrt",
              "--create-vrts/--no-create-vrts",
              default=False,
              help="Create (or skip) creation of VRTs, grouping files of like bands. Default is to skip VRT creation.")
@click.option("-nt",
              "--num-threads",
              type=int,
              default=8,
              help="Number of threads for gdalwarp. Default is 8.")
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
               num_threads,
               tmpdir):
    """Wrapper for orthorectification using GDAL utilities.

    This script assumes that the necessary utilities are accessible
    from the execution environment.

    """

    # Initialize logging (root level WARNING, app level INFO)
    logging_format = "[%(asctime)s|%(levelname)s|%(name)s|%(lineno)d] %(message)s"
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=logging_format)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Convert the input pixel size to the target spatial reference
    # system. It's easier to specify the pixel size in meters even
    # though the target SRS might require some other unit.
    try:
        # Set up a coordinate transformation from Web Mercator (which
        # uses meters) to the target projection.
        wm_ref = osr.SpatialReference()
        wm_ref.ImportFromEPSG(3857)
        tgt_ref = osr.SpatialReference()
        tgt_ref.SetFromUserInput(str(target_srs))
        xform_to_wm = osr.CoordinateTransformation(tgt_ref, wm_ref)
        xform_to_tgt = osr.CoordinateTransformation(wm_ref, tgt_ref)

        # Transform the target SRS origin into web mercator
        origin = xform_to_wm.TransformPoint(0.0, 0.0)

        # Add the pixel size in meters to the origin. Use the Y
        # coordinate because UTM zones are weird and their center is
        # actually at x=500000. That could introduce error if the
        # target SRS is a UTM zone.
        pt = xform_to_tgt.TransformPoint(origin[0], origin[1] + pixel_size)

        # The desired target SRS resolution is the point minus the
        # origin, i.e. pt[1] - 0.0
        pixel_size_srs = pt[1]
        logger.info("Target SRS '%s' pixel size is %.10f" % \
                    (target_srs, pixel_size_srs))

    finally:
        wm_ref = None
        tgt_ref = None
        xform_to_wm = None
        xform_to_tgt = None

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

    # Create a working directory for temporary data
    temp_dir = tempfile.mkdtemp(dir=tmpdir)
    output_files_by_band = defaultdict(list)
    try:
        # Loop over inputs
        for (input_file, info) in input_files.iteritems():
            logger.info("Processing %s" % input_file)
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(output_dir,
                                       os.path.relpath(input_file, input_dir))

            # Scale the pixel size according to this image's GSD
            # compared to the maximum GSD of all the images.
            scale_ratio = info.avg_gsd / min_gsd
            this_pixel_size = pixel_size * scale_ratio
            this_pixel_size_srs = pixel_size_srs * scale_ratio

            # Check DEM input
            if rpc_dem:
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
                           str(GDAL_CACHE_SIZE_MB),
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

                # Check whether the DEM needs to be adjusted to height above geoid (MSL)
                if apply_geoid:
                    # Subset geoid to match the DEM chip
                    geoid_chip = os.path.join(temp_dir, base_name + "_GEOID.tif")
                    logger.info("Subsetting geoid, (lat, lon) = (%.10f, %.10f) - (%.10f, %.10f)" % \
                                (min_lat, min_lon, max_lat, max_lon))
                    __run_cmd(["gdalwarp",
                               "--config",
                               "GDAL_CACHEMAX",
                               str(GDAL_CACHE_SIZE_MB),
                               "-wm",
                               str(WARP_CACHE_SIZE_MB),
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
                            (target_srs, this_pixel_size))
                if not os.path.isdir(output_file_dir):
                    os.makedirs(output_file_dir)
                __run_cmd(["gdalwarp",
                           "--config",
                           "GDAL_CACHEMAX",
                           str(GDAL_CACHE_SIZE_MB),
                           "-wm",
                           str(WARP_CACHE_SIZE_MB),
                           "-t_srs",
                           str(target_srs),
                           "-rpc",
                           "-tr",
                           str(this_pixel_size_srs),
                           str(this_pixel_size_srs),
                           "-r",
                           str(resampling_method),
                           "-multi",
                           "-wo",
                           "NUM_THREADS=%s" % num_threads,
                           "-to",
                           "RPC_DEM=%s" % dem_chip,
                           "-to",
                           "RPC_DEMINTERPOLATION=cubic",
                           "-co",
                           "TILED=YES",
                           input_file,
                           output_file],
                          fail_msg="Failed to orthorectify %s using DEM %s" % \
                          (input_file, dem_chip),
                          cwd=temp_dir)

            else:
                # No DEM, use strip average elevation. Find the
                # average lat/lon.
                avg_lat = (info.min_lat + info.max_lat) / 2.0
                avg_lon = (info.min_lon + info.max_lon) / 2.0

                # Get the geoid height at the average lat/lon. The
                # geoid file has no SRS but its geotransform matches
                # the lat/lon coordinate system, i.e. x=-180->+180,
                # y=-90->+90.
                logger.info("Getting geoid height at (lat, lon) = (%.10f, %.10f)" % \
                            (avg_lat, avg_lon))
                (stdout, stderr) = __run_cmd(["gdallocationinfo",
                                              "-valonly",
                                              "-b",
                                              "1",
                                              "-geoloc",
                                              GEOID_PATH,
                                              str(avg_lon),
                                              str(avg_lat)],
                                             fail_msg="Failed to get geoid height at (lat, lon) = (%.10f, %.10f)" % \
                                             (avg_lat, avg_lon),
                                             cwd=temp_dir,
                                             capture_streams=True)
                try:
                    geoid_height = float(stdout)
                except ValueError:
                    raise CommandError("gdallocationinfo returned invalid geoid height: "
                                       "stdout='%s', stderr='%s'" % (stdout, stderr))

                # Determine RPC height
                rpc_height = info.avg_hae + geoid_height
                logger.info("Using RPC height %.10f above geoid" % rpc_height)

                # Orthorectify
                output_file_dir = os.path.dirname(output_file)
                logger.info("Orthorectifying to SRS %s, %.5f meter pixels" % \
                            (target_srs, this_pixel_size))
                if not os.path.isdir(output_file_dir):
                    os.makedirs(output_file_dir)
                __run_cmd(["gdalwarp",
                           "--config",
                           "GDAL_CACHEMAX",
                           str(GDAL_CACHE_SIZE_MB),
                           "-wm",
                           str(WARP_CACHE_SIZE_MB),
                           "-t_srs",
                           str(target_srs),
                           "-rpc",
                           "-tr",
                           str(this_pixel_size_srs),
                           str(this_pixel_size_srs),
                           "-r",
                           str(resampling_method),
                           "-multi",
                           "-wo",
                           "NUM_THREADS=%s" % num_threads,
                           "-to",
                           "RPC_HEIGHT=%s" % rpc_height,
                           "-co",
                           "TILED=YES",
                           input_file,
                           output_file],
                          fail_msg="Failed to orthorectify %s using average height %.10f" % \
                          (input_file, rpc_height),
                          cwd=temp_dir)

            # Copy the input file's IMD to the output location
            shutil.copy(info.imd_file,
                        os.path.join(output_file_dir,
                                     os.path.basename(info.imd_file)))

            # Add the output file to the finished list
            output_files_by_band[info.band_id].append(output_file)

    finally:
        # Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)

    if create_vrts:
        # Create a VRT for each band
        for (band_id, output_files) in output_files_by_band.iteritems():
            # Check for a band alias to make a friendlier VRT name
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

def __get_imd_info(imd_file):
    """Parse an IMD file for metadata.

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

def __run_cmd(args, fail_msg="Command failed", cwd=None, env=None, capture_streams=False):
    # Get logger
    logger = logging.getLogger(__name__)

    # Convert args to strings
    string_args = [str(a) for a in args]
    logger.info("Running command: %s" % " ".join(['"%s"' % a for a in string_args]))

    # Set up streams (None means child process inherits from parent process)
    stdout_val = subprocess.PIPE if capture_streams else None
    stderr_val = subprocess.PIPE if capture_streams else None

    # Spawn child process and wait for it to complete
    p_obj = subprocess.Popen(string_args,
                             cwd=cwd,
                             env=env,
                             stdout=stdout_val,
                             stderr=stderr_val)
    (stdout, stderr) = p_obj.communicate()
    retval = p_obj.returncode
    if retval != 0:
        raise CommandError("%s, returned non-zero exit status %d" % \
                           (fail_msg, retval))
    return (stdout, stderr)

# Legacy entry point for calling script directly
if __name__ == "__main__":
    gdal_ortho()

