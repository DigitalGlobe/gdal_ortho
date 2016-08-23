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
import fiona
import fiona.crs
import fiona.transform
import shapely.geometry
import shapely.ops
from osgeo import osr

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
@click.argument("input_dir", type=click.Path(exists=True))
@click.argument("output_dir", type=click.Path())
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
              default=None,
              help="Path to DEM for orthorectification. (If omitted, the worldwide DEM stored in s3://dgdem is used. "
              "Contact the GBDX team for access to the bucket.)")
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

    # Fix paths
    input_dir = os.path.realpath(input_dir)
    output_dir = os.path.realpath(output_dir)
    if rpc_dem is not None:
        rpc_dem = os.path.realpath(rpc_dem)
    if tmpdir is not None:
        tmpdir = os.path.realpath(tmpdir)

    # Parse band list
    if bands is not None:
        bands_to_process = set(re.split(r"\s+|\s*,\s*", bands))
    else:
        bands_to_process = None

    # Override DEM input if using height above ellipsoid
    if use_hae:
        rpc_dem = None

    # Walk the input directory to find all the necessary files. Store
    # by part number then by band.
    part_shps = defaultdict(dict)
    part_dirs = defaultdict(dict)
    part_info = defaultdict(dict)
    for (path, dirs, files) in os.walk(input_dir, followlinks=True):
        # Look for GIS_FILES directory
        if os.path.basename(path).lower() == "gis_files":
            for f in files:
                # Find the per-part PIXEL_SHAPE files
                m_obj = re.search(r"\w+-(\w)\w+-\w+_p(\d+)_pixel_shape.shp$", f, flags=re.IGNORECASE)
                if m_obj is not None:
                    band_char = m_obj.group(1)
                    part_num = int(m_obj.group(2))
                    if band_char not in FILENAME_BAND_ALIASES:
                        logger.warn("PIXEL_SHAPE filename %s contains unknown band character %s" % \
                                    (f, band_char))
                    else:
                        band_alias = FILENAME_BAND_ALIASES[band_char]
                        part_shps[part_num][band_alias] = os.path.join(path, f)

        # Look for part directories
        m_obj = re.search(r".+_p(\d+)_\w+$", path, flags=re.IGNORECASE)
        if m_obj is not None:
            part_num = int(m_obj.group(1))

            # Look for IMD files
            imd_info = None
            for f in files:
                if os.path.splitext(f)[1].lower() == ".imd":
                    imd_info = __parse_imd(os.path.join(path, f))
            if imd_info is None:
                logger.warn("Part directory %s has no IMD file" % path)
            else:
                # Check band ID
                if imd_info.band_id not in IMD_BAND_ALIASES:
                    logger.warn("IMD file %s contains unknown bandId %s" % \
                                (imd_info.imd_file, imd_info.band_id))
                else:
                    band_alias = IMD_BAND_ALIASES[imd_info.band_id]
                    if bands_to_process is not None and \
                       band_alias not in bands_to_process:
                        logger.info("Skipping part directory %s (%s)" % \
                                    (path, band_alias))
                    else:
                        # Save this part directory
                        part_dirs[part_num][band_alias] = path
                        part_info[part_num][band_alias] = imd_info
                        logger.info("Found part directory %s (%s)" % \
                                    (path, band_alias))
    logger.info("Found %d part directories" % len(part_dirs))

    # Load all the shapefiles into one big geometry
    geoms = []
    for band_shps in part_shps.itervalues():
        for shp_filename in band_shps.itervalues():
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

    # Transform the full geometry into the target SRS. Its bounding
    # box defines the origin of the grid that each TIF should be
    # orthorectified into.
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
        # Loop over all the image info and find the best (smallest)
        # GSD. This will be used to define the pixel size in the
        # target SRS.
        min_gsd = min([info.avg_gsd
                       for band_info in part_info.itervalues()
                       for info in band_info.itervalues()])
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
        # transformed pixel as the pixel size. The larger dimension
        # will just end up being slightly oversampled, so no data is
        # lost.
        bounds = pix.bounds
        pixel_size = min(abs(bounds[2] - bounds[0]),
                         abs(bounds[3] - bounds[1]))
        logger.info("Calculated pixel size in target SRS is %.10f" % pixel_size)

    # Find average height above ellipsoid over all parts
    hae_vals = [info.avg_hae
                for band_info in part_info.itervalues()
                for info in band_info.itervalues()]
    if hae_vals:
        avg_hae = sum(hae_vals) / len(hae_vals)
    else:
        avg_hae = 0.0
    logger.info("Average height above ellipsoid is %.10f" % avg_hae)

    # Create a pool of worker threads. Each worker thread will call
    # out to GDAL utilities to do actual work.
    worker_pool = ThreadPoolExecutorWithCallback(max_workers=num_parallel)

    # Create a working directory for temporary data
    temp_dir = tempfile.mkdtemp(dir=tmpdir)
    try:
        # Check whether to download DEM data from S3
        if not use_hae and rpc_dem is None:
            # The use_hae flag is not set and no DEM path was
            # specified on the command line. Try to copy from S3.
            dem_vrt = os.path.join(temp_dir, "dgdem_" + str(uuid.uuid4()) + ".vrt")
            if dem.fetch_dgdem_tiles(full_geom.buffer(DEM_FETCH_MARGIN_DEG), dem_vrt):
                logger.info("Downloaded DEM tiles, using VRT %s" % dem_vrt)
                rpc_dem = dem_vrt
            else:
                logger.warn("Failed to download DEM tiles from S3, reverting to "
                            "average height above ellipsoid")
                use_hae = True

        # Loop over parts and submit jobs to worker pool
        for part_num in part_dirs.iterkeys():
            # Extract bands for this part
            band_shps = part_shps[part_num]
            band_dirs = part_dirs[part_num]
            band_info = part_info[part_num]

            # Submit job
            worker_pool.submit(worker_thread,
                               part_num,
                               band_dirs,
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

    finally:
        # Delete the temporary directory and its contents
        shutil.rmtree(temp_dir)

def worker_thread(part_num,
                  band_dirs,
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
        part_num: Part number.
        band_dirs: Dictionary containing directory paths for each band
            in the part.
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
    for (band, band_input_dir) in band_dirs.iteritems():
        logger.info("Processing part P%03d band %s" % (part_num, band))
        shp_filename = band_shps[band]
        imd_filename = band_info[band].imd_file
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
                logger.info("Part P%03d band %s does not intersect AOI" % \
                            (part_num, band))
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

        # Find all TIFs in the input directory
        tif_list = [os.path.join(band_input_dir, f)
                    for f in os.listdir(band_input_dir)
                    if f.lower().endswith(".tif")]

        # Check whether DEM is available
        if rpc_dem is not None:
            # Get the DEM chip if it hasn't already been created
            if dem_chip is None:
                dem_chip = dem.local_dem_chip(os.path.basename(band_input_dir),
                                              temp_dir,
                                              band_geom,
                                              rpc_dem,
                                              apply_geoid,
                                              gdal_cachemax,
                                              warp_memsize,
                                              warp_threads,
                                              DEM_CHIP_MARGIN_DEG)

            # Orthorectify all TIFs in the input directory
            for input_file in tif_list:
                logger.info("Orthorectifying %s to SRS %s using pixel size %.10f" % \
                            (input_file, target_srs, band_pixel_size))

                # Get path of TIF relative to input_dir. This provides
                # the path below output_dir to use for the output
                # file.
                tif_rel_path = os.path.relpath(input_file, input_dir)
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
                args += [input_file]
                args += [output_file]
                run_cmd(args,
                        fail_msg="Failed to orthorectify %s using DEM %s" % \
                        (input_file, dem_chip),
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

        else: # rpc_dem is None
            # Orthorectify all TIFs in the input directory using
            # average height above ellipsoid
            for input_file in tif_list:
                logger.info("Orthorectifying %s to SRS %s using pixel size %.10f" % \
                            (input_file, target_srs, band_pixel_size))

                # Get path of TIF relative to input_dir. This provides
                # the path below output_dir to use for the output
                # file.
                tif_rel_path = os.path.relpath(input_file, input_dir)
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
                args += [input_file]
                args += [output_file]
                run_cmd(args,
                        fail_msg="Failed to orthorectify %s using average height %.10f" % \
                        (input_file, avg_hae),
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
