import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from collections import defaultdict, namedtuple
from concurrent.futures import ThreadPoolExecutor

import click
import shapely.geometry
import shapely.ops
import fiona
import fiona.crs
import fiona.transform
from osgeo import osr

import aws
import dem
from run_cmd import run_cmd

class InputError(Exception): pass

# Constants
DEM_FETCH_MARGIN_DEG = 0.30
DEM_CHIP_MARGIN_DEG = 0.25
UTM_ZONES_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "data",
                              "UTM_Zone_Boundaries.geojson")
IMD_BAND_ALIASES = {
    # IMD bandId: supported "bands" input value
    "P":     "PAN",
    "Multi": "MS",
    "MS1":   "MS",
    "MS2":   "MS",
    "All-S": "SWIR"
}
FILENAME_BAND_ALIASES = {
    # Filename char: supported "bands" input value
    "P": "PAN",
    "M": "MS",
    "A": "SWIR"
}

# Initialize logging (root level WARNING, app level INFO)
logging_format = "[%(asctime)s|%(levelname)s|%(name)s|%(lineno)d] %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.WARNING, format=logging_format)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

@click.command()
@click.option("-aoi",
              "--aoi",
              required=True,
              type=float,
              nargs=4,
              help="Area of interest (min_lon min_lat max_lon max_lat) to project.")
@click.option("-srs",
              "--srs",
              required=True,
              type=str,
              help="Target SRS to project AOI into.")
def aoi_to_srs(aoi, srs):
    """Converts a lat/lon bounding box to a target SRS."""

    # Handle special "UTM" target SRS
    if srs.lower() == "utm":
        utm_epsg_code = __get_utm_epsg_code((aoi[0] + aoi[2]) / 2.0,
                                            (aoi[1] + aoi[3]) / 2.0)
        srs = "EPSG:%d" % utm_epsg_code

    # Log inputs
    logger.info("Input bounds: %.10f %.10f %.10f %.10f" % \
                (aoi[0], aoi[1], aoi[2], aoi[3]))
    logger.info("Target SRS: %s" % srs)

    # Create a PROJ.4 string of the target SRS for easy fiona calls
    try:
        ref = osr.SpatialReference()
        ref.SetFromUserInput(str(srs))
        srs_proj4 = ref.ExportToProj4()
    finally:
        ref = None

    # Create a shapely geometry representing the AOI
    aoi_geom = shapely.geometry.Polygon([(aoi[0], aoi[1]),  # LL
                                         (aoi[0], aoi[3]),  # UL
                                         (aoi[2], aoi[3]),  # UR
                                         (aoi[2], aoi[1]),  # LR
                                         (aoi[0], aoi[1])]) # LL

    # Transform the geometry into the target SRS
    src_crs = fiona.crs.from_epsg(4326)
    dst_crs = fiona.crs.from_string(srs_proj4)
    aoi_geom_srs = shapely.geometry.mapping(aoi_geom)
    aoi_geom_srs = fiona.transform.transform_geom(src_crs,
                                                  dst_crs,
                                                  aoi_geom_srs)
    aoi_geom_srs = shapely.geometry.shape(aoi_geom_srs)

    # Report the bounds
    bounds = aoi_geom_srs.bounds
    logger.info("Output bounds: %.10f %.10f %.10f %.10f" % \
                (bounds[0], bounds[1], bounds[2], bounds[3]))

@click.command()
@click.argument("input_dir", type=str)
@click.argument("output_dir", type=str)
@click.option("-srs",
              "--target-srs",
              type=str,
              required=True,
              help="Target SRS for gdalwarp. Specify 'UTM' to auto-determine the EPSG code for the input's UTM zone.")
@click.option("-ps",
              "--pixel-size",
              type=float,
              default=None,
              help="Pixel resolution in units of the target SRS. (If omitted, native resolution is used.)")
@click.option("-aoi",
              "--aoi",
              type=float,
              nargs=4,
              default=(),
              help="Area of interest (min_x min_y max_x max_y) to orthorectify in units of the target SRS. (If "
              "omitted, the full bounds of the input are used.)")
@click.option("-b",
              "--bands",
              type=str,
              default=None,
              help="Comma-separated list of bands to process. Use identifiers PAN, MS, SWIR, e.g. 'PAN,MS'. (If "
              "omitted, all bands are used.)")
@click.option("-dem",
              "--rpc-dem",
              type=str,
              default="s3://dgdem/current.txt",
              help="Local path or S3 URL of DEM for orthorectification. (If omitted, the worldwide DEM stored in "
              "s3://dgdem is used. Contact the GBDX team for access to the bucket.)")
@click.option("-hae/-nohae",
              "--use-hae/--no-use-hae",
              default=False,
              help="Use image average height above ellipsoid instead of a DEM for orthorectification. Default is "
              "to use DEM.")
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
@click.option("-et",
              "--error-threshold",
              type=float,
              default=0.01,
              help="Error threshold in pixels for gdalwarp. Set to 0 to use the exact transformer. Default is "
              "0.01 pixels.")
@click.option("-vrt/-novrt",
              "--create-vrts/--no-create-vrts",
              default=False,
              help="Create (or skip) creation of VRTs, grouping files of like bands. Default is to skip VRT creation.")
@click.option("-cm",
              "--gdal-cachemax",
              type=click.IntRange(1, 2**20),
              default=8192,
              help="Per-process memory size in MB for GDAL cache. Default is 8192 MB.")
@click.option("-wm",
              "--warp-memsize",
              type=click.IntRange(1, 2**20),
              default=2048,
              help="Per-process memory size in MB for warping. Default is 2048 MB.")
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
              help="Local path for temporary files. Default is OS-specific.")
def gdal_ortho(input_dir,
               output_dir,
               target_srs,
               pixel_size,
               aoi,
               bands,
               rpc_dem,
               use_hae,
               apply_geoid,
               resampling_method,
               error_threshold,
               create_vrts,
               gdal_cachemax,
               warp_memsize,
               warp_threads,
               num_parallel,
               tmpdir):
    """Wrapper for orthorectification using GDAL utilities.

    This script orthorectifies a DigitalGlobe 1B product using the
    provided RPCs. The input directory should contain a single 1B
    product.

    This script assumes that the necessary utilities are accessible
    from the execution environment.

    """

    # Create a working directory for temporary data
    if tmpdir is not None:
        tmpdir = os.path.realpath(tmpdir)
    temp_dir = tempfile.mkdtemp(dir=tmpdir)
    try:
        # Handle S3 input
        if aws.is_s3_url(input_dir):
            s3_input_prefix = input_dir
            if not s3_input_prefix.endswith("/"):
                s3_input_prefix += "/"
            input_dir = tempfile.mkdtemp(dir=temp_dir)
            run_cmd(["aws", "s3", "sync", s3_input_prefix, input_dir],
                    fail_msg="Failed to download input %s to %s" % \
                    (s3_input_prefix, input_dir),
                    cwd=input_dir)

        # Handle S3 output
        s3_output_prefix = None
        if aws.is_s3_url(output_dir):
            s3_output_prefix = output_dir
            if not s3_output_prefix.endswith("/"):
                s3_output_prefix += "/"
            output_dir = tempfile.mkdtemp(dir=temp_dir)

        # Fix paths
        input_dir = os.path.realpath(input_dir)
        output_dir = os.path.realpath(output_dir)
        if os.path.exists(rpc_dem):
            rpc_dem = os.path.realpath(rpc_dem)

        # Parse band list
        if bands is not None:
            bands_to_process = set(re.split(r"\s+|\s*,\s*", bands))
        else:
            bands_to_process = None

        # Walk the input directory to find IMD files and shapefiles
        imd_paths = {}
        shp_paths = {}
        for (path, dirs, files) in os.walk(input_dir, followlinks=True):
            # Look for GIS_FILES directory
            if os.path.basename(path).lower() == "gis_files":
                for f in files:
                    # Find PIXEL_SHAPE shapefiles
                    m_obj = re.search(r"^(.+)_pixel_shape.shp$", f, flags=re.IGNORECASE)
                    if m_obj is not None:
                        shp_paths[m_obj.group(1)] = os.path.join(path, f)
            else:
                # Look for IMDs
                for f in files:
                    if f.lower().endswith(".imd"):
                        imd_paths[os.path.splitext(f)[0]] = os.path.join(path, f)

        # Parse IMDs to determine the bands in each file
        imd_infos = defaultdict(list)
        for (base_name, imd_file) in imd_paths.iteritems():
            imd_info = __parse_imd(imd_file)
            if imd_info.band_id not in IMD_BAND_ALIASES:
                logger.warn("IMD file %s contains unknown bandId %s" % \
                            (imd_info.imd_file, imd_info.band_id))
            else:
                band_alias = IMD_BAND_ALIASES[imd_info.band_id]
                if bands_to_process is not None and \
                   band_alias not in bands_to_process:
                    logger.info("Skipping %s (%s)" % (base_name, band_alias))
                else:
                    # This IMD belongs to an image to be processed.
                    # Group by basename without respect to the bands
                    # in the filename (i.e. group corresponding P1BS
                    # and M1BS together).
                    imd_infos[__get_general_basename(base_name)].append(imd_info)
        logger.info("Found %d images to orthorectify" % len(imd_infos))

        # Load all the shapefiles into one big geometry
        geoms = []
        for shp_filename in shp_paths.itervalues():
            with fiona.open(shp_filename, "r") as shp:
                geoms += [shapely.geometry.shape(rec["geometry"]) for rec in shp]
        full_geom = shapely.ops.unary_union(geoms)

        # Handle special "UTM" target SRS
        utm_epsg_code = None
        if target_srs.lower() == "utm":
            utm_epsg_code = __get_utm_epsg_code(full_geom.centroid.y,
                                                full_geom.centroid.x)
            target_srs = "EPSG:%d" % utm_epsg_code
            logger.info("UTM target SRS is %s" % target_srs)

        # Create a PROJ.4 string of the target SRS for easy fiona calls
        try:
            srs = osr.SpatialReference()
            srs.SetFromUserInput(str(target_srs))
            target_srs_proj4 = srs.ExportToProj4()
        finally:
            srs = None

        # Transform the full geometry into the target SRS. Its
        # bounding box defines the origin of the grid that each TIF
        # should be orthorectified into.
        src_crs = fiona.crs.from_epsg(4326)
        dst_crs = fiona.crs.from_string(target_srs_proj4)
        full_geom_srs = shapely.geometry.mapping(full_geom)
        full_geom_srs = fiona.transform.transform_geom(src_crs,
                                                       dst_crs,
                                                       full_geom_srs)
        full_geom_srs = shapely.geometry.shape(full_geom_srs)
        grid_origin = full_geom_srs.bounds[0:2]
        logger.info("Ortho grid origin: %.10f, %.10f" % \
                    (grid_origin[0], grid_origin[1]))

        # Check whether pixel_size needs to be calculated
        if pixel_size is None:
            # Loop over all the image info and find the best
            # (smallest) GSD. This will be used to define the pixel
            # size in the target SRS.
            min_gsd = min([imd_info.avg_gsd
                           for imd_info_list in imd_infos.itervalues()
                           for imd_info in imd_info_list])
            logger.info("Best input GSD is %.10f" % min_gsd)

            # Get the UTM zone to use
            if utm_epsg_code is None:
                utm_epsg_code = __get_utm_epsg_code(full_geom.centroid.y,
                                                    full_geom.centroid.x)

            # Transform the full geometry's centroid into UTM
            src_crs = fiona.crs.from_epsg(4326)
            dst_crs = fiona.crs.from_epsg(utm_epsg_code)
            pt = shapely.geometry.mapping(full_geom.centroid)
            pt = fiona.transform.transform_geom(src_crs, dst_crs, pt)
            pt = shapely.geometry.shape(pt)

            # Add the best GSD to define a square in UTM space
            pix = shapely.geometry.box(pt.x, pt.y, pt.x + min_gsd, pt.y + min_gsd)

            # Transform the pixel box into the target SRS
            src_crs = dst_crs
            dst_crs = fiona.crs.from_string(target_srs_proj4)
            pix = shapely.geometry.mapping(pix)
            pix = fiona.transform.transform_geom(src_crs, dst_crs, pix)
            pix = shapely.geometry.shape(pix)

            # Use the smaller dimension from the bounding box of the
            # transformed pixel as the pixel size. The larger
            # dimension will just end up being slightly oversampled,
            # so no data is lost.
            bounds = pix.bounds
            pixel_size = min(abs(bounds[2] - bounds[0]),
                             abs(bounds[3] - bounds[1]))
            logger.info("Calculated pixel size in target SRS is %.10f" % pixel_size)

        # Find average height above ellipsoid over all parts
        hae_vals = [imd_info.avg_hae
                    for imd_info_list in imd_infos.itervalues()
                    for imd_info in imd_info_list]
        if hae_vals:
            avg_hae = sum(hae_vals) / len(hae_vals)
        else:
            avg_hae = 0.0
        logger.info("Average height above ellipsoid is %.10f" % avg_hae)

        # Create a pool of worker threads. Each worker thread will
        # call out to GDAL utilities to do actual work.
        worker_pool = ThreadPoolExecutorWithCallback(max_workers=num_parallel)

        # Check whether to download DEM data from S3
        if not use_hae and not os.path.exists(rpc_dem):
            # The use_hae flag is not set and the DEM path doesn't
            # exist locally. Assume it is an S3 path.
            dem_vrt = os.path.join(temp_dir, "s3dem_" + str(uuid.uuid4()) + ".vrt")
            if dem.download_tiles(rpc_dem,
                                  full_geom.buffer(DEM_FETCH_MARGIN_DEG),
                                  dem_vrt):
                logger.info("Downloaded DEM tiles, using VRT %s" % dem_vrt)
                rpc_dem = dem_vrt
            else:
                logger.warn("Failed to download DEM tiles from S3, reverting to "
                            "average height above ellipsoid")
                use_hae = True

        # Loop over images and submit jobs to worker pool
        for (gen_base_name, imd_info_list) in imd_infos.iteritems():
            # Get inputs by band
            band_info = {}
            band_shps = {}
            for imd_info in imd_info_list:
                imd_base_name = os.path.splitext(os.path.basename(imd_info.imd_file))[0]
                if imd_base_name not in shp_paths:
                    logger.warn("Base name %s missing from GIS_FILES" % imd_base_name)
                else:
                    band_info[imd_info.band_id] = imd_info
                    band_shps[imd_info.band_id] = shp_paths[imd_base_name]

            # Submit job
            worker_pool.submit(worker_thread,
                               gen_base_name,
                               band_info,
                               band_shps,
                               target_srs,
                               target_srs_proj4,
                               grid_origin,
                               pixel_size,
                               aoi,
                               rpc_dem,
                               avg_hae,
                               apply_geoid,
                               resampling_method,
                               error_threshold,
                               gdal_cachemax,
                               warp_memsize,
                               warp_threads,
                               input_dir,
                               output_dir,
                               temp_dir)

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

        # Create VRTs if requested
        if create_vrts and not canceled:
            # Walk output directory looking for TIFs
            tifs_by_band = defaultdict(list)
            for (path, dirs, files) in os.walk(output_dir, followlinks=True):
                for f in files:
                    m_obj = re.search(r"\w+-(\w)\w+-\w+_p\d+.tif$", f, flags=re.IGNORECASE)
                    if m_obj is not None:
                        band_char = m_obj.group(1)
                        if band_char not in FILENAME_BAND_ALIASES:
                            logger.warn("Output TIF filename %s contains unknown band character %s" % \
                                        (f, band_char))
                        else:
                            band_alias = FILENAME_BAND_ALIASES[band_char]
                            tifs_by_band[band_alias].append(os.path.join(path, f))

            # Create a VRT for each band
            for (band_alias, tif_list) in tifs_by_band.iteritems():
                # Use the first TIF's name
                m_obj = re.search(r"(.+)_p\d+.tif$",
                                  os.path.basename(tif_list[0]),
                                  flags=re.IGNORECASE)
                if m_obj is not None:
                    vrt_name = m_obj.group(1) + ".vrt"
                else:
                    vrt_name = "ortho_%s.vrt" % band_alias

                # Get relative paths to files from the output directory
                relpaths = [os.path.relpath(f, output_dir) for f in tif_list]

                # Create VRT (paths are relative to output_dir)
                logger.info("Creating band %s VRT %s" % (band_alias, vrt_name))
                run_cmd(["gdalbuildvrt",
                         "-srcnodata",
                         "0",
                         vrt_name] + relpaths,
                        fail_msg="Failed to create band %s VRT %s" % (band_alias, vrt_name),
                        cwd=output_dir)

        # Stage output to S3 if necessary
        if s3_output_prefix is not None:
            run_cmd(["aws", "s3", "sync", output_dir, s3_output_prefix],
                    fail_msg="Failed to upload output %s to %s" % \
                    (output_dir, s3_output_prefix),
                    cwd=output_dir)

    finally:
        # Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)

def worker_thread(base_name,
                  band_info,
                  band_shps,
                  target_srs,
                  target_srs_proj4,
                  grid_origin,
                  pixel_size,
                  aoi,
                  rpc_dem,
                  avg_hae,
                  apply_geoid,
                  resampling_method,
                  error_threshold,
                  gdal_cachemax,
                  warp_memsize,
                  warp_threads,
                  input_dir,
                  output_dir,
                  temp_dir):
    """Orthorectifies a 1B part using GDAL utilities.

    Args:
        base_name: Name of the image(s) to be processed.
        band_info: Tuples generated from __parse_imd for each band in
            the part.
        band_shps: Paths to shapefiles for each band in the part.
        target_srs: Spatial reference system to warp into.
        target_srs_proj4: PROJ.4 string of the target SRS.
        grid_origin: Tuple containing origin x and y of the ortho grid.
        pixel_size: Requested pixel size in target SRS units.
        aoi: Tuple containing min x, min y, max x, and max y bounds of
            AOI to orthorectify in target SRS units.
        rpc_dem: Path to DEM to use for warping.
        avg_hae: Average height above ellipsoid for all input images.
        apply_geoid: True to add geoid height to DEM, false to
            skip. Necessary for DEMs that are measured from
            geoid. (Most are.)
        resampling_method: Resampling method to use for warping.
        error_threshold: Error threshold in pixels for gdalwarp.
        gdal_cachemax: Cache size to use for GDAL utilities.
        warp_memsize: Extra cache size for warping.
        warp_threads: Number of threads to use for warping.
        input_dir: Path to base input directory.
        output_dir: Path to base output directory.
        temp_dir: Path to scratch directory for intermediate files.

    """

    # Determine per-band pixel sizes
    min_gsd = min([info.avg_gsd for info in band_info.itervalues()])
    band_pixel_sizes = {}
    for (band, info) in band_info.iteritems():
        band_pixel_sizes[band] = pixel_size * round(info.avg_gsd / min_gsd)

    # Loop over bands
    dem_chip = None
    for (band, info) in band_info.iteritems():
        logger.info("Processing %s %s" % (base_name, band))
        shp_filename = band_shps[band]
        imd_filename = info.imd_file
        xml_filename = os.path.splitext(imd_filename)[0] + ".XML"
        band_pixel_size = band_pixel_sizes[band]

        # Read in the shapefile
        with fiona.open(shp_filename, "r") as shp:
            recs = [shapely.geometry.shape(rec["geometry"]) for rec in shp]
            band_geom = shapely.ops.unary_union(recs)

        # Transform the geometry into the target SRS
        src_crs = fiona.crs.from_epsg(4326)
        dst_crs = fiona.crs.from_string(target_srs_proj4)
        band_geom_srs = shapely.geometry.mapping(band_geom)
        band_geom_srs = fiona.transform.transform_geom(src_crs,
                                                       dst_crs,
                                                       band_geom_srs)
        band_geom_srs = shapely.geometry.shape(band_geom_srs)

        # If an AOI was provided, intersect it with the geometry to
        # subset the output image
        if aoi:
            # Create a shapely geometry representing the AOI. The
            # coordinates are in the target SRS already.
            aoi_geom = shapely.geometry.Polygon([(aoi[0], aoi[1]),  # LL
                                                 (aoi[0], aoi[3]),  # UL
                                                 (aoi[2], aoi[3]),  # UR
                                                 (aoi[2], aoi[1]),  # LR
                                                 (aoi[0], aoi[1])]) # LL

            # Intersect the AOI with the full geometry. If there is no
            # intersection, there's no work to be done for this part.
            band_geom_srs = band_geom_srs.intersection(aoi_geom)
            if band_geom_srs.area == 0:
                logger.info("%s %s does not intersect AOI" % \
                            (base_name, band))
                continue

        # Calculate the extents to use given the ortho grid
        # origin. This ensures that all images are orthorectified into
        # an aligned grid. Expand the geometry bounds to the nearest
        # pixel, then calculate the extents as pixel offsets from the
        # origin.
        bounds = band_geom_srs.bounds
        min_pix_x = math.floor((bounds[0] - grid_origin[0]) / band_pixel_size)
        min_pix_y = math.floor((bounds[1] - grid_origin[1]) / band_pixel_size)
        max_pix_x = math.ceil((bounds[2] - grid_origin[0]) / band_pixel_size)
        max_pix_y = math.ceil((bounds[3] - grid_origin[1]) / band_pixel_size)
        min_extent_x = grid_origin[0] + (min_pix_x * band_pixel_size)
        min_extent_y = grid_origin[1] + (min_pix_y * band_pixel_size)
        max_extent_x = grid_origin[0] + (max_pix_x * band_pixel_size)
        max_extent_y = grid_origin[1] + (max_pix_y * band_pixel_size)

        # Find the TIF corresponding to the IMD
        input_dir = os.path.dirname(imd_filename)
        imd_basename = os.path.splitext(os.path.basename(imd_filename))[0]
        tif_list = [os.path.join(input_dir, f)
                    for f in os.listdir(input_dir)
                    if f.lower().endswith(".tif") and \
                    os.path.splitext(os.path.basename(f))[0] == imd_basename]
        if len(tif_list) != 1:
            logger.warn("Found %d TIFs corresponding to IMD file %s, expected 1" % \
                        (len(tif_list), imd_filename))
        else:
            tif_filename = tif_list[0]

            # Check whether DEM is available
            if os.path.exists(rpc_dem):
                # Get the DEM chip if it hasn't already been created
                if dem_chip is None:
                    dem_chip = dem.local_dem_chip(base_name,
                                                  temp_dir,
                                                  band_geom,
                                                  rpc_dem,
                                                  apply_geoid,
                                                  gdal_cachemax,
                                                  warp_memsize,
                                                  warp_threads,
                                                  DEM_CHIP_MARGIN_DEG)

                # Orthorectify TIF
                logger.info("Orthorectifying %s to SRS %s using pixel size %.10f" % \
                            (tif_filename, target_srs, band_pixel_size))

                # Get path of TIF relative to input_dir. This provides
                # the path below output_dir to use for the output
                # file.
                tif_rel_path = os.path.relpath(tif_filename, input_dir)
                output_file = __update_filename(os.path.join(output_dir, tif_rel_path))
                output_file_dir = os.path.dirname(output_file)
                if not os.path.isdir(output_file_dir):
                    os.makedirs(output_file_dir)
                args = ["gdalwarp"]
                args += ["--config", "GDAL_CACHEMAX", str(gdal_cachemax)]
                args += ["-wm", str(warp_memsize)]
                args += ["-t_srs", str(target_srs)]
                args += ["-rpc"]
                args += ["-te", str(min_extent_x), str(min_extent_y), str(max_extent_x), str(max_extent_y)]
                args += ["-tr", str(band_pixel_size), str(band_pixel_size)]
                args += ["-r", str(resampling_method)]
                args += ["-et", str(error_threshold)]
                args += ["-multi"]
                args += ["-wo", "NUM_THREADS=%s" % warp_threads]
                args += ["-to", "RPC_DEM=%s" % dem_chip]
                args += ["-to", "RPC_DEMINTERPOLATION=bilinear"]
                args += ["-co", "TILED=YES"]
                args += [tif_filename]
                args += [output_file]
                run_cmd(args,
                        fail_msg="Failed to orthorectify %s using DEM %s" % \
                        (tif_filename, dem_chip),
                        cwd=temp_dir)

                # Copy the input file's IMD to the output
                # location. Also copy the corresponding XML file if it
                # exists.
                updated_imd_filename = __update_filename(os.path.join(output_file_dir,
                                                                      os.path.basename(imd_filename)))
                shutil.copy(imd_filename, updated_imd_filename)
                __update_product_level(updated_imd_filename)
                if os.path.isfile(xml_filename):
                    updated_xml_filename = __update_filename(os.path.join(output_file_dir,
                                                                          os.path.basename(xml_filename)))
                    shutil.copy(xml_filename, updated_xml_filename)
                    __update_product_level(updated_xml_filename)

            else: # rpc_dem does not exist
                # Orthorectify TIF using average height above ellipsoid
                logger.info("Orthorectifying %s to SRS %s using pixel size %.10f" % \
                            (tif_filename, target_srs, band_pixel_size))

                # Get path of TIF relative to input_dir. This provides
                # the path below output_dir to use for the output
                # file.
                tif_rel_path = os.path.relpath(tif_filename, input_dir)
                output_file = __update_filename(os.path.join(output_dir, tif_rel_path))
                output_file_dir = os.path.dirname(output_file)
                if not os.path.isdir(output_file_dir):
                    os.makedirs(output_file_dir)
                args = ["gdalwarp"]
                args += ["--config", "GDAL_CACHEMAX", str(gdal_cachemax)]
                args += ["-wm", str(warp_memsize)]
                args += ["-t_srs", str(target_srs)]
                args += ["-rpc"]
                args += ["-te", str(min_extent_x), str(min_extent_y), str(max_extent_x), str(max_extent_y)]
                args += ["-tr", str(band_pixel_size), str(band_pixel_size)]
                args += ["-r", str(resampling_method)]
                args += ["-et", str(error_threshold)]
                args += ["-multi"]
                args += ["-wo", "NUM_THREADS=%s" % warp_threads]
                args += ["-to", "RPC_HEIGHT=%s" % avg_hae]
                args += ["-co", "TILED=YES"]
                args += [tif_filename]
                args += [output_file]
                run_cmd(args,
                        fail_msg="Failed to orthorectify %s using average height %.10f" % \
                        (tif_filename, avg_hae),
                        cwd=temp_dir)

                # Copy the input file's IMD to the output
                # location. Also copy the corresponding XML file if it
                # exists.
                shutil.copy(imd_filename,
                            __update_filename(os.path.join(output_file_dir,
                                                           os.path.basename(imd_filename))))
                if os.path.isfile(xml_filename):
                    shutil.copy(xml_filename,
                                __update_filename(os.path.join(output_file_dir,
                                                               os.path.basename(xml_filename))))

def __get_general_basename(filename):
    """Returns the basename of a file without band info.

    Args:
        filename: Filename to be parsed.

    Returns the basename of the provided filename with out band
    information, e.g. M1BS, P1BS, etc.

    """

    base_name = os.path.splitext(os.path.basename(filename))[0]
    m_obj = re.search(r"[_\-](\w1bs)[_\-]", base_name, flags=re.IGNORECASE)
    if m_obj is not None:
        base_name = base_name.replace(m_obj.group(1), "x" * len(m_obj.group(1)))
    return base_name

def __parse_imd(imd_file):

    """Parses an IMD file for metadata.

    Args:
        imd_file: Path to IMD file to be parsed.

    Returns a namedtuple with the following fields (NOTE: each field
    is stored as a string):
        imd_file: Path to IMD file.
        band_id: Band identifier reported by IMD file.
        avg_hae: Average height above ellipsoid of image.
        avg_gsd: Average ground sample distance of image.

    """

    # Create return type
    InfoType = namedtuple("InfoType",
                          ["imd_file",
                           "band_id",
                           "avg_hae",
                           "avg_gsd"])

    # Read IMD contents
    with open(imd_file, "r") as f_obj:
        imd_str = f_obj.read()

    # Create a dict of all the values to read
    params = {
        "bandId": [],
        "ULHAE": [],
        "URHAE": [],
        "LRHAE": [],
        "LLHAE": [],
        "meanProductGSD": []
    }

    # Read each value
    for (param_name, param_val_list) in params.iteritems():
        for m_obj in re.finditer(r'%s\s*=\s*"?([^";]+)"?;' % param_name, imd_str):
            param_val_list.append(str(m_obj.group(1)))

    # Find float values
    heights = [float(val) for param in ["ULHAE", "URHAE", "LRHAE", "LLHAE"] for val in params[param]]
    gsds = [float(val) for val in params["meanProductGSD"]]

    # Get band ID (should only be one)
    band_id = params["bandId"][0]

    return InfoType(imd_file=imd_file,
                    band_id=band_id,
                    avg_hae=sum(heights)/len(heights),
                    avg_gsd=sum(gsds)/len(gsds))

def __get_utm_epsg_code(lat, lon):
    """Looks up the UTM zone for a point.

    This function uses the UTM Zone Boundaries shapefile from this
    location:
    http://earth-info.nga.mil/GandG/coordsys/grids/universal_grid_system.html

    Args:
        lat: Latitude of the point to use.
        lon: Longitude of the point to use.

    Returns the integer EPSG code for the UTM zone containing the input point.

    """

    # Load the UTM zones
    zone_geoms = {}
    with open(UTM_ZONES_PATH, "r") as f_obj:
        zones_dict = json.load(f_obj)
    for feature in zones_dict["features"]:
        zone_geom = shapely.geometry.shape(feature["geometry"])
        zone_str = feature["properties"]["Zone_Hemi"]
        zone_geoms[zone_str] = zone_geom

    # Loop through zones and find the zone that contains the point
    pt = shapely.geometry.Point([lon, lat])
    found_zone = None
    for (zone_str, zone_geom) in zone_geoms.iteritems():
        if zone_geom.contains(pt):
            found_zone = zone_str
    if found_zone is None:
        raise InputError("Latitude %.10f Longitude %.10f is not in any UTM zone" % \
                         (lat, lon))

    # Parse the zone
    (zone_num, hemisphere) = found_zone.split(",")
    if hemisphere == "n":
        base_epsg = 32600
    else:
        base_epsg = 32700
    return base_epsg + int(zone_num)

def __update_filename(filename):
    """Updates the product level in a filename from 1B to 3X (custom ortho).

    Args:
        filename: Filename to update.

    Returns the updated filename.

    """

    m_obj = re.search(r"(.+-\w)1B(\w-.+)", filename)
    if m_obj is not None:
        return m_obj.group(1) + "3X" + m_obj.group(2)
    return filename

def __update_product_level(filename):
    """Updates the product level in metadata files from 1B to 3X (custom ortho).

    Args:
        filename: Filename of file to update in place.

    """

    ext = os.path.splitext(filename)[1].lower()
    if ext == ".imd":
        with open(filename, "r+") as f_obj:
            file_str = f_obj.read()
            file_str = re.sub(r'(productLevel\s*=\s*")LV1B(";)',
                              r"\1LV3X\2",
                              file_str)
            f_obj.seek(0)
            f_obj.write(file_str)

    elif ext == ".xml":
        with open(filename, "r+") as f_obj:
            file_str = f_obj.read()
            file_str = re.sub(r"(<PRODUCTLEVEL>)LV1B(</PRODUCTLEVEL>)",
                              r"\1LV3X\2",
                              file_str)
            f_obj.seek(0)
            f_obj.write(file_str)

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
