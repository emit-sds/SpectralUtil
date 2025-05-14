from spectral.io import envi
from osgeo import gdal
import netCDF4 as nc
import os
import numpy as np
import logging

numpy_to_gdal = {
    np.dtype(np.float64): 7,
    np.dtype(np.float32): 6,
    np.dtype(np.int32): 5,
    np.dtype(np.uint32): 4,
    np.dtype(np.int16): 3,
    np.dtype(np.uint16): 2,
    np.dtype(np.uint8): 1,
}

class GenericGeoMetadata:
    def __init__(self, band_names, geotransform=None, projection=None, glt=None, pre_orthod=False, nodata_value=None, loc=None):
        """
        Initializes the GenericGeoMetadata object.

        Args:
            band_names (list): List of band names.
            geotransform (tuple, optional): GDAL-style geotransform array. Defaults to None.
            projection (str, optional): Projection string. Defaults to None.
            glt (numpy.ndarray, optional): 3d array of x and y indices. Defaults to None.
            pre_orthod (bool, optional): Whether the data is already orthod. Defaults to False.
            nodata_value (int, optional): The nodata value. Defaults to None.
        """
        self.band_names = band_names
        self.geotransform = geotransform
        self.projection = projection
        self.glt = glt
        self.pre_orthod = False
        self.nodata_value = nodata_value
        self.loc = loc

        if pre_orthod:
            self.orthoable = False
        elif self.glt is None:
            self.orthoable = False
        else:
            self.orthoable = True


class SpectralMetadata:
    def __init__(self, wavelengths, fwhm, geotransform=None, projection=None, glt=None, pre_orthod=False, nodata_value=None):
        """
        Initializes the SpectralMetadata object.

        Args:
            wavelengths (numpy.ndarray): Array of wavelength values.
            fwhm (numpy.ndarray): Array of full-width half-maximum values.
            geotransform (tuple, optional): GDAL-style geotransform array. Defaults to None.
            projection (str, optional): Projection string. Defaults to None.
            glt (numpy.ndarray, optional): 3d array of x and y indices. Defaults to None.
            pre_orthod (bool, optional): Whether the data is already orthod. Defaults to False.
            nodata_value (int, optional): The nodata value. Defaults to None.
        """
        self.wavelengths = wavelengths
        self.wl = wavelengths
        self.fwhm = fwhm
        self.geotransform = geotransform
        self.projection = projection
        self.glt = glt
        self.pre_orthod = False
        self.nodata_value = nodata_value

        if pre_orthod:
            self.orthoable = False
        elif self.glt is None:
            self.orthoable = False
        else:
            self.orthoable = True

    def wl_index(self, wl, buffer=None):
        """
        Finds the index of the wavelength closest to the given wavelength.

        Args:
            wl (float): The wavelength to find the index for.
            buffer (float, optional): The spectral range around the center wavelength to include. Defaults to None.

        Returns:
            int or numpy.ndarray: The index of the closest wavelength if buffer is None or 0. 
                      If buffer is provided, returns an array of indices within the buffer range.
        """
        if buffer is None or buffer == 0:
            return np.argmin(np.abs(self.wl - wl))
        else:
            return np.where(np.logical_and(self.wl >= wl - buffer, self.wl <= wl + buffer))


def load_data(input_file, lazy=True, load_glt=False, load_loc=False):
    """
    Loads a file and extracts the spectral metadata and data.

    Args:
        input_file (str): Path to the input file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.
        load_glt (bool, optional): If True, loads the glt for orthoing. Defaults to False.

    Raises:
        ValueError: If the file type is unknown.

    Returns:
        tuple: A tuple containing:
            - Metadata: An object containing the appropriate metadata
            - numpy.ndarray or netCDF4.Variable: The data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f'{input_file} not found.')

    input_filename = os.path.basename(input_file)
    if input_filename.endswith(('.hdr', '.dat', '.img')) or '.' not in input_filename:
        return open_envi(input_file, lazy=lazy)
    elif input_filename.endswith('.nc'):
        return open_netcdf(input_file, lazy=lazy, load_glt=load_glt, load_loc=load_loc)
    elif input_filename.endswith('.tif') or input_filename.endswith('.vrt'):
        return open_tif(input_file, lazy=lazy)
    else:
        raise ValueError(f'Unknown file type for {input_file}')

def ortho_data(data, glt, glt_mask=None, glt_nodata=0, nodata_value=-9999):
    """
    Orthorectifies the data using the provided ground control points.

    Args:
        data (numpy.ndarray): The spectral data to orthorectify, with shape (rows, cols, bands).
        glt (numpy.ndarray, optional): 3d array of x and y indices. Defaults to None. Negatives assumed as interpolation values.
        glt_mask (numpy.ndarray, optional): A mask to apply to the glt. Defaults to None.
        glt_nodata (int, optional): The nodata value for the glt. Defaults to 0.
        nodata_value (int, optional): The nodata value to fill in the background with.

    Returns:
        numpy.ndarray: The orthorectified data.
    """
    do_squeeze = False
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis = -1)
        do_squeeze = True

    outdata = np.zeros((glt.shape[0], glt.shape[1], data.shape[2]), dtype=data.dtype) + nodata_value
    valid_glt = np.all(glt != glt_nodata, axis=-1)
    if glt_mask is not None:
        valid_glt = np.logical_and(valid_glt, glt_mask)
    
    glt_tmp = np.abs(glt).astype(int)

    if glt_nodata == 0:
        glt_tmp[valid_glt] -= 1
    outdata[valid_glt, :] = data[np.abs(glt_tmp[valid_glt, 1]), np.abs(glt_tmp[valid_glt, 0]), :]

    if do_squeeze:
        data = np.squeeze(data) # Change it back!
        outdata = np.squeeze(outdata)

    return outdata


def write_cog(output_file, data, meta, ortho=True, nodata_value=-9999):
    """
    Writes a Cloud Optimized GeoTIFF (COG) file.

    Args:
        output_file (str): Path to the output COG file.
        data (numpy.ndarray): The spectral data to be written, with shape (rows, cols, bands).
        meta (SpectralMetadata): The spectral metadata containing geotransform and projection information.
        ortho (bool, optional): Whether to ortho the data.  Only relevant if the data isn't natively orthod. Defaults to True.
        nodata_value (, optional): The nodata value to use. Defaults to -9999.
        gdal_dtype (int, optional): The GDAL data type to use. Defaults to 6 (Float32).
    """
    driver = gdal.GetDriverByName('MEM')

    if ortho and meta.orthoable:
        od = ortho_data(data, meta.glt, nodata_value=nodata_value)
    else:
        od = data

    ds = driver.Create('', od.shape[1], od.shape[0], od.shape[2], numpy_to_gdal[od.dtype])
    if meta.geotransform is not None:
        ds.SetGeoTransform(meta.geotransform)
    if meta.projection is not None:
        ds.SetProjection(meta.projection)
    for i in range(od.shape[2]):
        ds.GetRasterBand(i+1).WriteArray(od[:, :, i])
        ds.GetRasterBand(i+1).SetNoDataValue(nodata_value)
        #if meta.band_names is not None:
        #    ds.GetRasterBand(i+1).SetDescription(meta.band_names[i])
    ds.BuildOverviews('NEAREST', [2, 4, 8, 16, 32, 64, 128])
    
    tiff_driver = gdal.GetDriverByName('GTiff')
    output_dataset = tiff_driver.CreateCopy(output_file, ds, options=['COMPRESS=LZW', 'BIGTIFF=YES','COPY_SRC_OVERVIEWS=YES', 'TILED=YES', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'])
    ds = None
    output_dataset = None


def envi_header(input_file):
    """
    Generates the corresponding ENVI header file path for a given input file.

    Args:
        input_file (str): Path to the input file.

    Returns:
        str: Path to the corresponding ENVI header file.
    """
    return os.path.splitext(input_file)[0] + '.hdr'


def open_envi(input_file, lazy=True):
    """
    Opens an ENVI file and extracts the spectral metadata and data.

    Args:
        input_file (str): Path to the ENVI file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The data, either as a lazy-loaded memory map or a fully loaded numpy array.
    """
    header = envi_header(input_file)
    ds = envi.open(header)
    imeta = ds.metadata
    if 'wavelength' in imeta:
        wl = np.array([float(x) for x in imeta['wavelength']])
    else:
        wl = None
    if 'fwhm' in imeta:
        fwhm = np.array([float(x) for x in imeta['fwhm']])
    else:
        fwhm = None
    if 'data ignore value' in imeta:
        nodata_value = float(imeta['data ignore value'])
    else:
        nodata_value = -9999 # set default

    if 'coordinate system string' in imeta:
        css = imeta['coordinate system string']
        proj = css if type(css) == str else ','.join(css)
    else:
        proj = None
    if 'map info' in imeta:
        map_info = imeta['map info'].split(',')
        trans = [float(map_info[3]), float(map_info[5]), 0, float(map_info[4]), 0, -float(map_info[6])]
    else:
        map_info, trans = None, None

    if lazy:
        rfl = ds.open_memmap(interleave='bip', writable=False)
    else:
        rfl = ds.open_memmap(interleave='bip').copy()

    meta = SpectralMetadata(wl, fwhm, nodata_value=nodata_value, geotransform=trans, projection=proj)
    return meta, rfl
    

def open_tif(input_file, lazy=False):
    """
    Opens a GeoTIFF file and extracts the spectral metadata and data.

    Args:
        input_file (str): Path to the GeoTIFF file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - GenericGeoMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray: The data, either as a lazy-loaded memory map or a fully loaded numpy array.
    """
    if lazy:
        logging.warning('Lazy loading not supported for GeoTIFF data.')
    ds = gdal.Open(input_file)
    proj = ds.GetProjection()
    trans = ds.GetGeoTransform()
    band_names = [ds.GetRasterBand(i+1).GetDescription() for i in range(ds.RasterCount)]
    nodata_value = ds.GetRasterBand(1).GetNoDataValue()
    meta = GenericGeoMetadata(band_names, geotransform=trans, projection=proj, pre_orthod=True, nodata_value=nodata_value)
    data = ds.ReadAsArray()
    if len(data.shape) == 3:
        data = np.transpose(data, (1, 2, 0))
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    return meta, data


def open_netcdf(input_file, lazy=True, load_glt=False, load_loc=False):
    """
    Opens a NetCDF file and extracts the metadata and data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the data lazily. Defaults to True.
        load_glt (bool, optional): If True, loads the glt for orthoing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - Metadata: An object containing the appropriate metadata
            - numpy.ndarray or netCDF4.Variable: The data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    input_filename = os.path.basename(input_file)
    if 'EMIT' in input_filename and 'RAD' in input_filename:
        return open_emit_rdn(input_file, lazy=lazy, load_glt=load_glt)
    elif 'AV3' in input_filename and 'RFL' in input_filename:
        return open_airborne_rfl(input_file, lazy=lazy)
    elif 'AV3' in input_filename and 'RDN' in input_filename:
        return open_airborne_rdn(input_file, lazy=lazy)
    elif ('av3' in input_filename.lower() or 'ang' in input_filename.lower()) and 'OBS' in input_filename:
        return open_airborne_obs(input_file, lazy=lazy, load_glt=load_glt, load_loc=load_loc)
    elif 'ang' in input_filename.lower()  and 'rfl' in input_filename.lower():
        return open_airborne_rfl(input_file, lazy=lazy)
    else:
        raise ValueError(f'Unknown file type for {input_file}')


def open_emit_rdn(input_file, lazy=True, load_glt=False):
    """
    Opens an EMIT radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the radiance data lazily. Defaults to True.
        load_glt (bool, optional): If True, loads the glt for orthoing. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The radiance data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    ds = nc.Dataset(input_file)
    wl = ds['sensor_band_parameters']['wavelengths'][:]
    fwhm = ds['sensor_band_parameters']['fwhm'][:]
    trans = ds.geotransform
    proj = ds.spatial_ref
    nodata_value = float(ds['radiance']._FillValue)

    if lazy:
        rdn = ds['radiance']
    else:
        rdn = np.array(ds['radiance'][:])
    
    glt = None
    if load_glt:
        glt = np.stack([ds['location']['glt_x'][:],ds['location']['glt_y'][:]],axis=-1)

    meta = SpectralMetadata(wl, fwhm, trans, proj, glt, pre_orthod=False, nodata_value=nodata_value)

    return meta, rdn


def open_airborne_rfl(input_file, lazy=True):
    """
    Opens an Airborne reflectance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the reflectance data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The reflectance data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    ds = nc.Dataset(input_file)
    wl = ds['reflectance']['wavelength'][:]
    fwhm = ds['reflectance']['fwhm'][:]
    proj = ds.variables['transverse_mercator'].spatial_ref
    trans = [float(x) for x in ds.variables['transverse_mercator'].GeoTransform.split(' ')]
    nodata_value = float(ds['reflectance']['reflectance']._FillValue)

    if lazy:
        # This is too bad....we're forced into inconsistent handling between AV3 and EMIT
        # need to consider some clever solutions.  In the meantime, this works, but is expensive
        rfl = np.transpose(ds['reflectance']['reflectance'], (1,2,0))
    else:
        rfl = np.transpose(ds['reflectance']['reflectance'][:], (1,2,0))
    
    meta = SpectralMetadata(wl, fwhm, trans, proj, glt=None, pre_orthod=True, nodata_value=nodata_value)

    return meta, rfl

def open_airborne_rdn(input_file, lazy=True):
    """
    Opens an Airborne radiance NetCDF file and extracts the spectral metadata and radiance data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the radiance data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The radiance data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    ds = nc.Dataset(input_file)
    wl = ds['radiance']['wavelength'][:]
    fwhm = ds['radiance']['fwhm'][:]
    proj = ds.variables['transverse_mercator'].spatial_ref
    trans = [float(x) for x in ds.variables['transverse_mercator'].GeoTransform.split(' ')]
    nodata_value = float(ds['radiance']['radiance']._FillValue)

    if lazy:
        # This is too bad....we're forced into inconsistent handling between AV3 and EMIT
        # need to consider some clever solutions.  In the meantime, this works, but is expensive
        rdn = np.transpose(ds['radiance']['radiance'], (1,2,0))
    else:
        rdn = np.transpose(ds['radiance']['radiance'][:], (1,2,0))
    
    meta = SpectralMetadata(wl, fwhm, trans, proj, glt=None, pre_orthod=True, nodata_value=nodata_value)

    return meta, rdn




def open_airborne_obs(input_file, lazy=True, load_glt=False, load_loc=False):
    """
    Opens an Airborne observation NetCDF file and extracts the spectral metadata and obs data.

    Args:
        input_file (str): Path to the NetCDF file.
        lazy (bool, optional): If True, loads the radiance data lazily. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - SpectralMetadata: An object containing the wavelengths and FWHM.
            - numpy.ndarray or netCDF4.Variable: The radiance data, either as a lazy-loaded variable or a fully loaded numpy array.
    """
    ds = nc.Dataset(input_file)
    proj = ds.variables['transverse_mercator'].spatial_ref
    trans = [float(x) for x in ds.variables['transverse_mercator'].GeoTransform.split(' ')]


    obs_names = list(ds['observation_parameters'].variables.keys())

    nodata_value = float(ds['observation_parameters'][obs_names[0]]._FillValue)
    glt = None
    if load_glt:
        glt = np.stack([ds['geolocation_lookup_table']['sample'][:],ds['geolocation_lookup_table']['line'][:]],axis=-1)
    loc = None
    if load_loc:
        loc = np.stack([ds['lon'][:],ds['lat'][:]],axis=-1)

    # Don't have a good solution for lazy here, temporarily ignoring...
    if lazy:
        logging.warning("Lazy loading not supported for observation data.")
    obs = np.stack([ds['observation_parameters'][on] for on in obs_names], axis=-1)
    
    meta = GenericGeoMetadata(obs_names, trans, proj, glt=glt, pre_orthod=True, nodata_value=nodata_value, loc=loc)

    return meta, obs


def get_extent_from_obs(input_file, get_resolution=False):
    """
    Gets the extent of the observation data.

    Args:
        input_file (str): Path to the input file.
        get_resolution (bool, optional): If True, returns the resolution as well. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - float: Upper left x-coordinate.
            - float: Upper left y-coordinate.
            - float: Lower right x-coordinate.
            - float: Lower right y-coordinate.
    """
    # replace with open_airborne_obs once the lazy loading works
    ds = nc.Dataset(input_file)
    lat = ds['lat'][:]
    lon = ds['lon'][:]
    if get_resolution:
        # Not quite right
        res = np.abs(np.mean(np.diff(lat))), np.abs(np.mean(np.diff(lon)))
        return np.min(lon), np.max(lat), np.max(lon), np.min(lat), None, None
    else:
        return np.min(lon), np.max(lat), np.max(lon), np.min(lat)

def write_envi_file(dat, meta, output_filename, interleave = 'BIL'):
    """
    Write an ENVI file
    Args:
        dat: data to write: nx, ny, nbands
        meta (SpectralMetadata): The spectral metadata contianing wavelengths and FWHM
        output_filename: Output file name (no extension)
        interleave: string to indicate interleave method
    """
    if interleave.lower() != 'bil':
        raise NotImplementedError(f'Print interleave mode {interleave} not yet supported.')

    create_envi_file(output_filename, dat.shape, meta, dtype = dat.dtype)
    write_bil_chunk(dat.transpose([0,2,1]), output_filename, 0, (dat.shape[0], dat.shape[2], dat.shape[1]))

def write_bil_chunk(dat, outfile, line, shape, dtype = 'float32'):
    """
    Write a chunk of data to a binary, BIL formatted data cube.
    Args:
        dat: data to write
        outfile: output file to write to
        line: line of the output file to write to
        shape: shape of the output file
        dtype: output data type

    Returns:
        None
    """
    outfile = open(outfile, 'rb+')
    outfile.seek(line * shape[1] * shape[2] * np.dtype(dtype).itemsize)
    outfile.write(dat.astype(dtype).tobytes())
    outfile.close()


def create_envi_file(output_file, data_shape, meta, dtype='float32'):
    """
    Creates an ENVI file with the given data and metadata.

    Args:
        output_file (str): Path to the output ENVI file.
        data_shape (tuple): The shape of the data to be written (rows, cols, bands).
        meta (SpectralMetadata): The spectral metadata containing wavelengths and FWHM.
        dtype (str, optional): The data type of the output file. Defaults to 'float32'.
    """

    # Create base file....gdal doesn't seem to like to write all the metadata, so we'll clean that up after
    driver = gdal.GetDriverByName('ENVI')
    driver.Register()
    outDataset = driver.Create(output_file, data_shape[1], data_shape[0], data_shape[2], numpy_to_gdal[dtype], options=['INTERLEAVE=BIL'])
    outDataset.SetGeoTransform(meta.geotransform)
    outDataset.SetProjection(meta.projection)
    del outDataset


    # Touch up the header file
    header = envi.read_envi_header(envi_header(output_file))

    if 'wl' in meta.__dict__ and meta.wl is not None:
        header['wavelength'] = '{ ' + ', '.join(map(str, meta.wl)) + ' }'
    if 'fwhm' in meta.__dict__ and meta.fwhm is not None:
        header['fwhm'] = '{ ' + ', '.join(map(str, meta.fwhm)) + ' }'
    if 'band_names' in meta.__dict__ and meta.band_names is not None:
        header['band names'] = '{ ' + ', '.join(meta.band_names) + ' }'

    header['data ignore value'] = str(meta.nodata_value)

    envi.write_envi_header(envi_header(output_file), header) 
