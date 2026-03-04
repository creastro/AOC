import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq, fftshift, ifftshift

from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip, sigma_clipped_stats, mad_std
from astropy.modeling import Fittable1DModel, Parameter

from photutils.centroids import centroid_quadratic
from photutils.psf import MoffatPSF, GaussianPSF, AiryDiskPSF

from scipy.special import j1

class Airy1D(Fittable1DModel):
    """
    One-dimensional Airy disk intensity model.

    This model represents the diffraction pattern of a circular aperture
    projected along one dimension.

    Model equation:
        I(x) = I0 * [2 J1(k (x - x0)) / (k (x - x0))]^2 + b

    Parameters
    ----------
    I0 : float
        Peak intensity.
    k : float
        Spatial frequency scaling factor.
    x0 : float
        Central position.
    b : float
        Constant background offset.

    Notes
    -----
    - J1 is the first-order Bessel function of the first kind.
    - The singularity at r = 0 is handled analytically.
    """
    
    I0 = Parameter(default=1.0)
    k = Parameter(default=1.0)
    x0 = Parameter(default=0.0)
    b = Parameter(default=0.0)

    @staticmethod
    def evaluate(x, I0, k, x0, b):
        r = k * (x - x0)
        # avoid division by zero at center
        y = np.ones_like(r)
        mask = r != 0
        y[mask] = (2 * j1(r[mask]) / r[mask])**2
        return I0 * y + b


def fit_gauss_airy_1d(r, y):
    """
    Fit a 1D radial profile using a compound Gaussian + Airy model.

    Parameters
    ----------
    r : ndarray
        Radial coordinate array (in pixels).

    y : ndarray
        Observed intensity profile.
        
    Returns
    -------
    fitted_model : CompoundModel
        Best-fit Astropy compound model (Gaussian1D + Airy1D).

    Notes
    -----
    - Fitting is performed using TRFLSQFitter (Trust Region Reflective).
    - Parameter bounds and fixed values can be applied before fitting.

    Examples
    --------
    model = fit_gauss_airy_1d(r, y)

    # Evaluate fitted profile
    fitted_profile = model(r)

    # Extract Gaussian
    gauss_amp   = model.amplitude_0.value
    gauss_sigma = model.stddev_0.value

    # Extract Airy
    airy_I0 = model.I0_1.value
    airy_k  = model.k_1.value
    """

    # Gaussian
    gauss = models.Gaussian1D(
        amplitude=y.max(),
        mean=0.0,
        stddev=1.0
    )

    # How to apply constraints:
    # -------------------------------
    # Fix parameter:
    # gauss.mean.fixed = True
    #
    # Apply bounds:
    # gauss.amplitude.bounds = (0, 2 * y.max())

    # Airy
    airy = Airy1D(
        I0=y.max() / 2,
        k=1.0,
        x0=0.0,
        b=np.median(y)
    )

    # Combine model
    model = gauss + airy

    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, r, y)

    return fitted_model


def fit_airy_moffat_1d(r, y):
    """
    Fit a 1D radial profile using a compound Moffat + Airy model.

    Parameters
    ----------
    r : ndarray
        Radial coordinate array (in pixels).

    y : ndarray
        Observed intensity profile.

    Returns
    -------
    fitted_model : CompoundModel
        Best-fit Astropy compound model (Moffat1D + Airy1D).

    Notes
    -----
    - Fitting is performed using TRFLSQFitter.

    Examples
    --------
    model = fit_moffat_airy_1d(r, y)

    fitted_profile = model(r)

    # Extract Moffat parameters
    moffat_amp   = model.amplitude_0.value
    moffat_gamma = model.gamma_0.value
    moffat_alpha = model.alpha_0.value

    # Extract Airy parameters
    airy_I0 = model.I0_1.value
    airy_k  = model.k_1.value
    """

    # Moffat
    moffat = models.Moffat1D(
        amplitude=y.max()*0.2,
        x_0=0.0,
        gamma=1.0,
        alpha=2.5
    )

    # Airy
    airy = Airy1D(
        I0=y.max()*0.8,
        k=1.0,
        x0=0.0,
        b=np.median(y)
    )

    airy.I0.bounds = (0, 2 * y.max())
    airy.x0.fixed = True
    # airy.k.bounds = (1e-4, 50)
    # airy.b.bounds = (0, y.max())
    
    # Combine model
    model = airy + moffat
    
    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, r, y)

    return fitted_model

def fit_gauss_moffat_1d(r, y):
    """
    Fit a 1D radial profile using a compound Gaussian + Moffat model.

    Parameters
    ----------
    r : ndarray
        Radial coordinate array (in pixels).

    y : ndarray
        Observed intensity profile.

    Returns
    -------
    fitted_model : CompoundModel
        Best-fit Astropy compound model (Gaussian1D + Moffat1D).

    Notes
    -----
    - Fitting is performed using TRFLSQFitter.

    Examples
    --------
    model = fit_gauss_moffat_1d(r, y)

    fitted_profile = model(r)

    # Extract Gaussian parameters
    gauss_amp   = model.amplitude_0.value
    gauss_sigma = model.stddev_0.value

    # Extract Moffat parameters
    moffat_amp   = model.amplitude_1.value
    moffat_gamma = model.gamma_1.value
    moffat_alpha = model.alpha_1.value
    """
    # Gaussian
    gauss = models.Gaussian1D(
        amplitude=0.8*y.max(),
        mean=0.0,
        stddev=1.0
    )

    gauss.amplitude.bounds = (0, 2 * y.max())
    gauss.mean.fixed = True

    # Moffat
    moffat = models.Moffat1D(
        amplitude=0.2 * y.max(),
        x_0=0.0,
        gamma=1.0,
        alpha=2.5
    )

    # Combine
    model = gauss + moffat

    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, r, y)

    return fitted_model
    

def fit_moffat_airy_2d(image, photometry=False):
    """
    Fit a 2D image using a compound Moffat + Airy model.

    Parameters
    ----------
    image : 2D ndarray
        Input image (ideally background-subtracted).

    photometry : bool, optional
        If True, use Photutils PSF models (flux-normalized).
        If False, use Astropy amplitude-based models.

    Returns
    -------
    fitted_model : CompoundModel
        Best-fit Moffat + Airy 2D model.

    Notes
    -----
    - TRFLSQFitter is used (Trust Region Reflective).
    - Image should be centered near peak.

    Examples
    --------
    y, x = np.indices(image.shape)
    model = fit_moffat_airy_2d(image)
    fitted_profile = model(x,y)
    """

    # Coordinate grid
    y, x = np.indices(image.shape)

    # Initial center guess
    y0, x0 = np.unravel_index(np.argmax(image), image.shape)

    amplitude = image.max()
    flux = np.sum(image)

    # Model initialization
    if photometry:
        # Photutils PSF models (flux-normalized)
        moffat = MoffatPSF(
            flux=0.2 * flux,
            x_0=x0,
            y_0=y0,
            alpha=6.0,
            beta=2.5
        )

        # moffat.alpha.bounds = (2.0, 20)
        # moffat.beta.bounds = (1.5, 10)

        airy = AiryDiskPSF(
            flux=0.8 * flux,
            x_0=x0,
            y_0=y0,
            radius=3.0
        )

        # airy.radius.bounds = (1.0, 20)

    else:
        # Astropy amplitude-based models
        moffat = models.Moffat2D(
            amplitude=0.2 * amplitude,
            x_0=x0,
            y_0=y0,
            gamma=6.0,
            alpha=2.5
        )

        # moffat.amplitude.bounds = (0, 2 * amplitude)
        # moffat.gamma.bounds = (1.0, 50)
        # moffat.alpha.bounds = (1.0, 10)

        airy = models.AiryDisk2D(
            amplitude=0.8 * amplitude,
            x_0=x0,
            y_0=y0,
            radius=4.0
        )

        airy.amplitude.bounds = (0, 2 * amplitude)
        airy.radius.bounds = (3.0, 10)
        # airy.x_0.fixed = True
        # airy.y_0.fixed = True

    # Combine model
    model = moffat + airy

    # Optional: tie centers
    # model.x_0_1.tied = lambda m: m.x_0_0
    # model.y_0_1.tied = lambda m: m.y_0_0

    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, x, y, image, filter_non_finite=True)

    return fitted_model

def fit_moffat_gauss_2d(image, photometry=False):
    """
    Fit a 2D image using a compound Moffat + Gaussian model.

    Parameters
    ----------
    image : 2D ndarray
        Input image (ideally background-subtracted).

    photometry : bool, optional
        If True, use Photutils PSF models (flux-normalized).
        If False, use Astropy amplitude-based models.

    Returns
    -------
    fitted_model : CompoundModel
        Best-fit Moffat + Gaussian 2D model.

    Notes
    -----
    - TRFLSQFitter is used (Trust Region Reflective).
    - Image should be centered near peak.

    Examples
    --------
    y, x = np.indices(image.shape)
    model = fit_moffat_gauss_2d(image)
    fitted_profile = model(x,y)
    """

    # Coordinate grid
    y, x = np.indices(image.shape)

    # Initial center guess
    y0, x0 = np.unravel_index(np.argmax(image), image.shape)

    sigma_x_init = np.std(np.mean(image, axis=0))
    sigma_y_init = np.std(np.mean(image, axis=1))

    amplitude = image.max()
    flux = np.sum(image)

    # Model initialization
    if photometry:
        # Photutils PSF models (flux-normalized)
        moffat = MoffatPSF(
            flux=0.2 * flux,
            x_0=x0,
            y_0=y0,
            alpha=6.0,
            beta=2.5
        )

        # moffat.alpha.bounds = (2.0, 20)
        # moffat.beta.bounds = (1.5, 10)
        
        gauss = GaussianPSF(
            flux=0.8 * flux,
            x_0=x0,
            y_0=y0,
            x_fwhm=3.0,
            y_fwhm=3.0
        )


    else:
        # Astropy amplitude-based models
        moffat = models.Moffat2D(
            amplitude=0.2 * amplitude,
            x_0=x0,
            y_0=y0,
            gamma=6.0,
            alpha=2.5
        )

        # moffat.amplitude.bounds = (0, 2 * amplitude)
        # moffat.gamma.bounds = (1.0, 50)
        # moffat.alpha.bounds = (1.0, 10)

        gauss = models.Gaussian2D(
            amplitude=amplitude,
            x_mean=x0,
            y_mean=y0,
            x_stddev=max(sigma_x_init, 1.0),
            y_stddev=max(sigma_y_init, 1.0)
        )

        gauss.amplitude.bounds = (0, 2 * amplitude)

    # Combine model
    model = moffat + gauss

    # Optional: tie centers
    # model.x_0_1.tied = lambda m: m.x_0_0
    # model.y_0_1.tied = lambda m: m.y_0_0

    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, x, y, image, filter_non_finite=True)

    return fitted_model

def fit_moffat2d(image, photometry=False):
    """
    Fit a 2D Moffat profile to the image.

    Parameters
    ----------
    image : 2D ndarray
        Input image (ideally background-subtracted).

    photometry : bool, optional
        If True, use Photutils MoffatPSF (flux-normalized).
        If False, use Astropy Moffat2D (amplitude-based).

    Returns
    -------
    fitted_model : Model
        Best-fit Moffat model.

    Notes
    -----
    - TRFLSQFitter (Trust Region Reflective) is used.
    - Image should be centered near the peak.
    Examples
    --------
    y, x = np.indices(image.shape)
    model = fit_moffat2d(image)
    fitted_profile = model(x,y)
    """

    # Coordinate grid
    y, x = np.indices(image.shape)

    # Initial guesses
    amplitude = image.max()
    y0, x0 = np.unravel_index(np.argmax(image), image.shape)

    if photometry:
        # Photutils flux-normalized model
        moffat = MoffatPSF(
            flux=image.sum(),
            x_0=x0,
            y_0=y0,
            alpha=2.5,
            beta=2.5
        )

        # moffat.alpha.bounds = (1.5, 20)
        # moffat.beta.bounds = (1.0, 10)

    else:
        # Astropy amplitude-based model
        moffat = models.Moffat2D(
            amplitude=amplitude,
            x_0=x0,
            y_0=y0,
            gamma=3.0,
            alpha=4.0
        )

        moffat.amplitude.bounds = (0, 2 * amplitude)
        # moffat.gamma.bounds = (0.5, 50)
        # moffat.alpha.bounds = (1.0, 10)

        # moffat.x_0.bounds = (x0 - 2, x0 + 2)
        # moffat.y_0.bounds = (y0 - 2, y0 + 2)

    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(moffat, x, y, image, filter_non_finite=True)

    return fitted_model

def fit_gaussian2d(image, photometry=False):
    """
    Fit a 2D Gaussian profile to the image.

    Parameters
    ----------
    image : 2D ndarray
        Input image (ideally background-subtracted).

    photometry : bool, optional
        If True, use Photutils CircularGaussianPSF (flux-normalized).
        If False, use Astropy Gaussian2D (amplitude-based).

    Returns
    -------
    fitted_model : Model
        Best-fit Gaussian model.

    Notes
    -----
    - TRFLSQFitter is used.
    - Image should be centered near the PSF peak.
    Examples
    --------
    y, x = np.indices(image.shape)
    model = fit_gaussian2d(image)
    fitted_profile = model(x,y)
    """

    # Coordinate grid
    ny, nx = image.shape
    y, x = np.indices(image.shape)
    
    # Initial guesses
    amplitude_init = image.max()
    y0_init, x0_init = np.unravel_index(np.argmax(image), image.shape)

    sigma_x_init = np.std(np.mean(image, axis=0))
    sigma_y_init = np.std(np.mean(image, axis=1))

    if photometry:
        # Photutils circular Gaussian (flux-based)
        model = CircularGaussianPSF(
            flux=image.sum(),
            x_0=x0_init,
            y_0=y0_init,
            fwhm=2.0
        )

        # model.fwhm.bounds = (0.5, 50)

    else:
        # Astropy elliptical Gaussian (amplitude-based)
        model = models.Gaussian2D(
            amplitude=amplitude_init,
            x_mean=x0_init,
            y_mean=y0_init,
            x_stddev=max(sigma_x_init, 1.0),
            y_stddev=max(sigma_y_init, 1.0)
        )

        model.amplitude.bounds = (0, 2 * amplitude_init)
        # model.x_stddev.bounds = (0.5, 50)
        # model.y_stddev.bounds = (0.5, 50)

        # model.x_mean.bounds = (x0_init - 2, x0_init + 2)
        # model.y_mean.bounds = (y0_init - 2, y0_init + 2)

    # Fit
    fitter = fitting.TRFLSQFitter()
    fitted_model = fitter(model, x, y, image, filter_non_finite=True)

    return fitted_model


def fft_recenter(image):
    """
    Recenter an image using FFT-based subpixel shifting.

    The shift is computed from the difference between the peak pixel
    and the photocenter (centroid).

    Parameters
    ----------
    image : 2D ndarray
        Input image.

    Returns
    -------
    shifted_image : 2D ndarray
        Recentered image with subpixel accuracy.

    Notes
    -----
    - Uses quadratic centroiding (Photutils).
    - Uses Fourier phase shifting.
    """

    image = np.asarray(image, dtype=float)
    ny, nx = image.shape

    # Peak pixel
    peak_y, peak_x = np.unravel_index(np.argmax(image), image.shape)

    total_flux = image.sum()
    if total_flux == 0:
        raise ValueError("Zero flux in image.")

    # Photocenter estimation
    mean, med, std = sigma_clipped_stats(image, stdfunc=mad_std)

    threshold = med + 3 * std
    mask = image < threshold

    pc_x, pc_y = centroid_quadratic(data=image, mask=mask)

    # Required subpixel shift
    dy = pc_y - (peak_y+0.5)
    dx = pc_x - (peak_x+0.5)

    # FFT-based shift
    ky = fftfreq(ny)
    kx = fftfreq(nx)
    kx, ky = np.meshgrid(kx, ky)

    phase = np.exp(-2j * np.pi * (kx * dx + ky * dy))
    shifted_image = ifft2(fft2(image) * phase).real

    return shifted_image




def fft_oversample2d(psf, factor=4):
    """
    Oversample a 2D PSF using Fourier zero-padding.

    Parameters
    ----------
    psf : 2D ndarray
        Input image (should be centered and background-subtracted).

    factor : int, optional
        Oversampling factor (default=4).

    Returns
    -------
    psf_os : 2D ndarray
        Oversampled image.
    """

    psf = np.asarray(psf, float)
    flux = psf.sum()

    if flux == 0:
        raise ValueError("Input image has zero total flux.")

    ny, nx = psf.shape
    ny2, nx2 = ny * factor, nx * factor

    # FFT
    F = fftshift(fft2(psf))

    # Zero-padded Fourier plane
    F2 = np.zeros((ny2, nx2), dtype=complex)

    cy, cx = ny // 2, nx // 2
    cy2, cx2 = ny2 // 2, nx2 // 2

    y1 = cy2 - cy
    y2 = y1 + ny
    x1 = cx2 - cx
    x2 = x1 + nx

    F2[y1:y2, x1:x2] = F

    # Inverse FFT
    psf_os = ifft2(ifftshift(F2)).real

    # Correct scaling (grid expansion effect)
    psf_os *= factor**2

    # Preserve total flux
    psf_os *= flux / psf_os.sum()

    return psf_os



def crop_img(psf, target_shape):
    """
    Crop a PSF image around its maximum pixel.

    Parameters
    ----------
    psf : 2D ndarray
        Input image.

    target_shape : tuple
        Desired output shape (ny, nx).

    Returns
    -------
    cropped_psf : 2D ndarray
        Cropped image centered on peak pixel.

    Notes
    -----
    - Cropping is centered on the maximum pixel.
    - Uses Astropy Cutout2D.
    - target_shape must be smaller than input shape.
    """

    psf = np.asarray(psf)

    ny_psf, nx_psf = psf.shape
    ny_t, nx_t = target_shape

    if ny_psf < ny_t or nx_psf < nx_t:
        raise ValueError("PSF must be larger than target shape.")

    # Locate peak pixel
    peak_y, peak_x = np.unravel_index(np.argmax(psf), psf.shape)

    # Create cutout (note: size = (ny, nx))
    cutout = Cutout2D(
        psf,
        position=(peak_x, peak_y),   # (x, y)
        size=(ny_t, nx_t)            # (ny, nx)
    )

    return cutout.data


def strehl_ratio(
    img,
    psf,
    aperture_radius_px=30
    ):

    img = np.asarray(img, float)
    psf_theory = np.asarray(psf, float)
    psf_theory = fft_oversample2d(psf_theory, factor = 4)

    if img.shape != psf_theory.shape:
        raise ValueError("Measured and theoretical PSFs must have same shape")

    ny, nx = img.shape
    cy, cx = ny / 2, nx / 2


    img = crop_img(img,(2*aperture_radius_px, 2*aperture_radius_px))

    
    psf_theory = crop_img(psf_theory,(2*aperture_radius_px, 2*aperture_radius_px))


    if oversampling:
        img_oversampled = fft_oversample2d(img, factor = factor)
        psf_oversampled = fft_oversample2d(psf_theory, factor = factor)
    else:
        img_oversampled = img
        psf_oversampled = psf_theory

    if recenter:
        img_oversampled = fft_recenter(img_oversampled)

    # --- Flux normalization
    F_meas = img_oversampled.sum()
    F_theory = psf_oversampled.sum()

    if F_meas <= 0 or F_theory <= 0:
        raise ValueError("Non-positive flux inside aperture")

    psf_meas_n = img_oversampled / F_meas
    psf_theory_n = psf_oversampled / F_theory

    # --- Strehl ratio
    strehl = psf_meas_n.max() / psf_theory_n.max()

    return strehl

def strehl_ratio(
    img,
    psf_theory,
    aperture_radius_px=30
):
    """
    Compute Strehl ratio between measured and theoretical PSFs.

    Parameters
    ----------
    img : 2D ndarray
        Measured PSF image.

    psf_theory : 2D ndarray
        Diffraction-limited theoretical PSF.

    aperture_radius_px : int
        Radius (in pixels) for flux normalization aperture.

    oversample_factor : int or None
        If provided, both PSFs are oversampled using FFT.

    recenter : bool
        If True, recenter both PSFs before computation.

    Returns
    -------
    strehl : float
        Estimated Strehl ratio.
    """

    img = np.asarray(img, float)
    psf_theory = np.asarray(psf_theory, float)


    # Crop around center
    size = 2 * aperture_radius_px

    img = crop_img(img, (size, size))
    psf_theory = crop_img(psf_theory, (size, size))


    if img.shape != psf_theory.shape:
        raise ValueError("Measured and theoretical PSFs must have same shape")

    # Flux normalization inside aperture
    F_meas = img.sum()
    F_theory = psf_theory.sum()

    if F_meas <= 0 or F_theory <= 0:
        raise ValueError("Non-positive flux inside aperture")

    psf_meas_n = img / F_meas
    psf_theory_n = psf_theory / F_theory

    # Strehl ratio
    strehl = psf_meas_n.max() / psf_theory_n.max()

    return strehl



def pupil2psf(
    pupil,
    pupil_pixsize_m = 1.04/240,
    psf_pixscale_mas = 135.22605304,
    wavelength_nm = 1600,
):
    """
    Compute the PSF from a pupil using physical units and zero padding.

    Parameters
    ----------
    pupil : 2D ndarray
        Pupil image. The pupil diameter must fill the array.
    pupil_pixsize_m : float
        Pupil pixel size in meters per pixel.
    psf_pixscale_mas : float
        Desired PSF angular sampling in milliarcseconds per pixel.
    wavelength_nm : float
        Wavelength in nanometers.

    Returns
    -------
    psf : 2D ndarray
        Focal-plane intensity image.
    psf_pixscale_mas : float
        Actual PSF pixel scale (mas/pix).
    """

    # --- Unit conversions ---
    wavelength_m = wavelength_nm * 1e-9
    mas_to_rad = np.pi / (180.0 * 3600.0 * 1000.0)
    psf_pixscale_rad = psf_pixscale_mas * mas_to_rad

    # --- Required FFT size ---
    N_fft = int(np.round(
        wavelength_m / (pupil_pixsize_m * psf_pixscale_rad)
    ))

    Dp = pupil.shape[0]

    if N_fft < Dp:
        raise ValueError(
            "Requested PSF pixel scale is too coarse for the given pupil sampling."
        )

    # --- Zero-padded pupil ---
    pupil_fft = np.zeros((N_fft, N_fft), dtype=np.complex128)

    y0 = (N_fft - Dp) // 2
    x0 = (N_fft - Dp) // 2
    pupil_fft[y0:y0+Dp, x0:x0+Dp] = pupil

    # --- FFT to focal plane ---
    field_fp = fftshift(fft2(ifftshift(pupil_fft)))
    intensity_fp = np.abs(field_fp)**2

    # --- Crop PSF to pupil size ---
    c0 = N_fft // 2
    h = Dp // 2
    psf = intensity_fp[c0 - h:c0 - h + Dp,
                       c0 - h:c0 - h + Dp]

    # Normalize
    psf /= psf.max()

    return psf

def radial_fwhm(r, I):
    """
    Compute FWHM of a radial profile via linear interpolation.

    Parameters
    ----------
    r : array_like
        Radial coordinates (monotonic increasing)
    I : array_like
        Radial profile values (peak at r[0])

    Returns
    -------
    fwhm : float
        Full width at half maximum
    """
    r = np.asarray(r)
    I = np.asarray(I)

    # Imax = I[0]
    Imax = I.max()
    half_max = Imax / 2.0

    # Find first index where profile drops below half max
    below = np.where(I <= half_max)[0]
    if len(below) == 0:
        raise ValueError("Profile never drops below half maximum")

    i2 = below[0]
    i1 = i2 - 1
    if i1 < 0:
        raise ValueError("Half-maximum occurs before first radial bin")

    # Linear interpolation
    r_half = r[i1] + (half_max - I[i1]) * (r[i2] - r[i1]) / (I[i2] - I[i1])
    fwhm = 2*r_half

    return fwhm

def fwhm_2d(image):
    """
    Estimate circular FWHM from area above half-maximum.

    Parameters
    ----------
    image : 2D ndarray
        Input image (background-subtracted).

    Returns
    -------
    fwhm : float
        Estimated circular FWHM (in pixels).

    Notes
    -----
    - Assumes circular symmetry.
    - Based on area above half-maximum.
    - Not suitable for highly asymmetric PSFs.
    """

    image = np.asarray(image, float)
    peak = image.max()
    halfmax = peak / 2
    mask = image >= halfmax
    area = np.sum(mask)
    fwhm = 2 * np.sqrt(area / np.pi)

    return fwhm

def savefits(
    image,
    header_source_fits,
    output_fits,
    ext=0,
    overwrite=True,
    dtype=None,
):
    """
    Save a 2D image to a FITS file using a header copied from another FITS file.

    Parameters
    ----------
    image : array-like
        2D image data to save.
    header_source_fits : str
        Path to FITS file to copy header from.
    output_fits : str
        Output FITS filename.
    ext : int or str, optional
        FITS extension to copy header from (default: 0).
    overwrite : bool, optional
        Overwrite output file if it exists.
    dtype : numpy dtype, optional
        Cast image to this dtype before saving (e.g. np.float32).
    """

    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("Input image must be 2D")

    if dtype is not None:
        image = image.astype(dtype)

    # Read header from source FITS
    with fits.open(header_source_fits) as hdul:
        header = hdul[ext].header.copy()

    # Remove keywords that may conflict with new data
    header.remove("BITPIX", ignore_missing=True)
    header.remove("NAXIS", ignore_missing=True)
    for key in list(header.keys()):
        if key.startswith("NAXIS"):
            header.remove(key, ignore_missing=True)

    # Create and write new FITS file
    hdu = fits.PrimaryHDU(data=image, header=header)
    hdu.writeto(output_fits, overwrite=overwrite)




    