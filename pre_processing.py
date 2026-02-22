import numpy as np
import scipy
import scipy.stats as ss
from scipy.ndimage import uniform_filter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def est_noise(y, noise_type='additive'):
    """
    This function infers the noise in a
    hyperspectral data set, by assuming that the
    reflectance at a given band is well modelled
    by a linear regression on the remaining bands.

    Parameters:
        y: numpy array
            a HSI cube ((m*n) x p)

       noise_type: string [optional 'additive'|'poisson']

    Returns: tuple numpy array, numpy array
        * the noise estimates for every pixel (N x p)
        * the noise correlation matrix estimates (p x p)

    Copyright:
        Jose Nascimento (zen@isel.pt) and Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    def est_additive_noise(r):
        small = 1e-6
        L, N = r.shape
        w=np.zeros((L,N), dtype=float)
        RR=np.dot(r,r.T)
        RRi = np.linalg.pinv(RR+small*np.eye(L))
        RRi = np.matrix(RRi)
        for i in range(L):
            XX = RRi - (RRi[:,i]*RRi[i,:]) / RRi[i,i]
            RRa = RR[:,i]
            RRa[i] = 0
            beta = np.dot(XX, RRa)
            beta[0,i]=0;
            w[i,:] = r[i,:] - np.dot(beta,r)
        Rw = np.diag(np.diag(np.dot(w,w.T) / N))
        return w, Rw

    y = y.T
    L, N = y.shape
    #verb = 'poisson'
    if noise_type == 'poisson':
        sqy = np.sqrt(y * (y > 0))
        u, Ru = est_additive_noise(sqy)
        x = (sqy - u)**2
        w = np.sqrt(x)*u*2
        Rw = np.dot(w,w.T) / N
    # additive
    else:
        w, Rw = est_additive_noise(y)
    return w.T, Rw.T


def hysime(y, n, Rn):
    """
    Hyperspectral signal subspace estimation

    Parameters:
        y: numpy array
            hyperspectral data set (each row is a pixel)
            with ((m*n) x p), where p is the number of bands
            and (m*n) the number of pixels.

        n: numpy array
            ((m*n) x p) matrix with the noise in each pixel.

        Rn: numpy array
            noise correlation matrix (p x p)

    Returns: tuple integer, numpy array
        * kf signal subspace dimension
        * Ek matrix which columns are the eigenvectors that span
          the signal subspace.

    Copyright:
        Jose Nascimento (zen@isel.pt) & Jose Bioucas-Dias (bioucas@lx.it.pt)
        For any comments contact the authors
    """
    y=y.T
    n=n.T
    Rn=Rn.T
    L, N = y.shape
    Ln, Nn = n.shape
    d1, d2 = Rn.shape

    x = y - n;

    Ry = np.dot(y, y.T) / N
    Rx = np.dot(x, x.T) / N
    E, dx, V = np.linalg.svd(Rx)

    Rn = Rn+np.sum(np.diag(Rx))/L/10**5 * np.eye(L)
    Py = np.diag(np.dot(E.T, np.dot(Ry,E)))
    Pn = np.diag(np.dot(E.T, np.dot(Rn,E)))
    cost_F = -Py + 2 * Pn
    kf = np.sum(cost_F < 0)
    ind_asc = np.argsort(cost_F)
    Ek = E[:, ind_asc[0:kf]]
    return kf, Ek # Ek.T ?



# Comments on using complex number:
#
# Use only scipy and numpy functions for a correct use of complex number.
#
# scipy.sqrt() deal by the book with complex number,
# it's more tricky when using math and numpy modules.
#


def HfcVd(M, far='default'):
    """
    Computes the vitual dimensionality (VD) measure for an HSI
    image for specified false alarm rates.  When no false alarm rate(s) is
    specificied, the following vector is used: 1e-3, 1e-4, 1e-5.
    This metric is used to estimate the number of materials in an HSI scene.

    Parameters:
       M: numpy array
           HSI data as a 2D matrix (N x p).

       far: list [default default]
           False alarm rate(s).

    Returns: python list
           VD measure, number of materials estimate.

    References:
        C.-I. Chang and Q. Du, "Estimation of number of spectrally distinct
        signal sources in hyperspectral imagery," IEEE Transactions on
        Geoscience and Remote Sensing, vol. 43, no. 3, mar 2004.

        J. Wang and C.-I. Chang, "Applications of independent component
        analysis in endmember extraction and abundance quantification for
        hyperspectral imagery," IEEE Transactions on Geoscience and Remote
        Sensing, vol. 44, no. 9, pp. 2601-1616, sep 2006.
    """
    N, numBands = M.shape

    # calculate eigenvalues of covariance and correlation between bands
    lambda_cov = np.linalg.eig(np.cov(M.T))[0] # octave: cov(M')
    lambda_corr = np.linalg.eig(np.corrcoef(M.T))[0] # octave: corrcoef(M')
    # not realy needed:
    lambda_cov = np.sort(lambda_cov)[::-1]
    lambda_corr = np.sort(lambda_corr)[::-1]

    if far == 'default':
        far = [10*-3, 10-4, 10*-5]
    else:
        far = [far]

    numEndmembers_list = []
    for y in range(len(far)):
        numEndmembers = 0
        pf = far[y]
        for x in range(numBands):
            sigmaSquared = (2.*lambda_cov[x]/N) + (2.*lambda_corr[x]/N) + (2./N)*lambda_cov[x]*lambda_corr[x]
            sigma = np.sqrt(sigmaSquared)
            tau = -ss.norm.ppf(pf, 0, abs(sigma))
            if (lambda_corr[x]-lambda_cov[x]) > tau:
                numEndmembers += 1
        numEndmembers_list.append(numEndmembers)
    return numEndmembers_list

def linear_spectral_unmixing(y, endmembers):
    """
    Linear Spectral Unmixing (LSU) to mitigate noise in hyperspectral data.

    Parameters:
        y: numpy array
            Hyperspectral data set ((m*n) x p).

        endmembers: numpy array
            Endmember matrix (p x q), where q is the number of endmembers.

    Returns:
        numpy array
            Noise-mitigated hyperspectral data.
    """
    # Estimate abundance fractions using least squares regression
    abundance_estimates = np.linalg.lstsq(endmembers, y.T, rcond=None)[0]

    # Reconstruct data using estimated abundances and endmembers
    reconstructed_data = np.dot(endmembers, abundance_estimates)

    return reconstructed_data.T  # Transpose to get back to original shape

def remove_noise(y, w):
    """
    Remove noise from hyperspectral data.

    Parameters:
        y: numpy array
            Original hyperspectral data.
        w: numpy array
            Noise estimates for every pixel.

    Returns:
        numpy array
            Noise-filtered hyperspectral data.
    """
    return y - w



def pre_call(white_ref,dark_ref,data_ref,id,num_components=6):
    white_nparr = np.array(white_ref.load())
    dark_nparr = np.array(dark_ref.load())
    data_nparr = np.array(data_ref.load())


    print(f"{'='*20}Working on {id}{'='*20}")
    print("\nRaw data shape:-\n White ref:{}\n Dark ref:{}\n Image:{}".format(white_nparr.shape,dark_nparr.shape,data_nparr.shape))

    corrected_nparr = 100*np.divide(np.subtract(data_nparr, dark_nparr),np.subtract(white_nparr, dark_nparr))
    x = 670
    y = 210
    spectral_signature = data_nparr[y, x, :]


    plt.plot(spectral_signature)
    plt.grid(1)
    plt.ylabel('Radiance')
    plt.xlabel('Wavelength (nm)')
    plt.savefig(f'/home/vs/HSI/results/{id}/pre_processing_output/Raw_spectral_signature.png',dpi=1000)
    plt.clf()

    plt.plot(corrected_nparr[y, x, :])
    plt.grid(1)
    plt.ylabel('Relative Reflectance')
    plt.xlabel('Wavelength (nm)')
    plt.savefig(f'/home/vs/HSI/results/{id}/pre_processing_output/Calibrated_spectral_signature.png',dpi=1000)
    plt.clf()

    hyperspectral_data =  corrected_nparr
    og_dim=corrected_nparr.shape
    print("\n\noriginal dimension:",og_dim)
    x_pixel, y_pixel, reflectance = hyperspectral_data.shape
    reduced_data = hyperspectral_data.reshape(x_pixel * y_pixel, reflectance)

    print("Reduced Hyperspectral Data Shape:", reduced_data.shape)

    print("starting with est_noise")
    hyperspectral_data =reduced_data
    noise_estimates, noise_corr_matrix = est_noise(hyperspectral_data)

    print("hysime initiated")
    kf, signal_subspace = hysime(hyperspectral_data, noise_estimates, noise_corr_matrix)
    print("hysime completed...HfcVd begins")
    vd_measure = HfcVd(hyperspectral_data)
    print("HfcVd done...noise is being removed")
    filtered_data = remove_noise(reduced_data,noise_estimates)
    n,m,z = og_dim
    post_HYSIME_data = filtered_data.reshape(n, m, z)
    plt.plot(post_HYSIME_data[y, x, :])
    plt.grid(1)
    plt.ylabel('Relative Reflectance')
    plt.xlabel('Wavelength (nm)')
    plt.savefig(f'/home/vs/HSI/results/{id}/pre_processing_output/HySIME_filtered_spectral_signature.png',dpi=1000)
    plt.clf()

    print("HySIME_plot_saved_complete")

    start_band = 751
    end_band = 826
    post_HYSIME_data_without_range = np.delete(post_HYSIME_data, np.s_[start_band:end_band+1], axis=2)

    start_band = 0
    end_band = 50
    post_HYSIME_data_without_range = np.delete(post_HYSIME_data_without_range, np.s_[start_band:end_band+1], axis=2)

    num_n_bands=140

    n_avg=post_HYSIME_data_without_range.shape[2]//num_n_bands

    x_pixel, y_pixel, reflectance =  post_HYSIME_data_without_range.shape
    reduced_data_physime = post_HYSIME_data_without_range.reshape(x_pixel * y_pixel, reflectance)

    hyperspectral_image=post_HYSIME_data_without_range
    desired_num_bands = 140
    total_bands = hyperspectral_image.shape[2]
    band_group_size = total_bands // desired_num_bands
    reshaped_image = hyperspectral_image.reshape(hyperspectral_image.shape[0], hyperspectral_image.shape[1], desired_num_bands, band_group_size)
    averaged_image = np.mean(reshaped_image, axis=3)
    print("Averaging done")

    window_size = 5
    smoothed_data = uniform_filter(averaged_image, size=(window_size, window_size, 1))
    for _ in range(50):
        smoothed_data = uniform_filter(smoothed_data, size=(window_size, window_size, 1))

    print("Smoothening complete")
    normalized_image=smoothed_data
    squared_image = normalized_image ** 2
    sum_of_squares = np.sqrt(np.sum(squared_image, axis=2))
    Y_BC = normalized_image / sum_of_squares[:, :, np.newaxis]

    print("normlisation complete")
    plt.plot(Y_BC[y, x, :])
    plt.grid(1)
    plt.ylabel('Normalised Reflectance')
    plt.xlabel('Feature')
    plt.savefig(f'/home/vs/HSI/results/{id}/pre_processing_output/Final_pre_processed_spectral_signature.png',dpi=1000)
    plt.clf()

    print(f"Applying PCA: Reducing 140 bands to {num_components} components...")

    # Flatten to (Pixels, 140) for PCA
    flat_data = Y_BC.reshape(-1, desired_num_bands)

    pca = PCA(n_components=num_components)
    pca_data = pca.fit_transform(flat_data)

    # Reshape back to (Height, Width, 6)
    final_preprocessed = pca_data.reshape(x_pixel, y_pixel, num_components)

    return {"preprocessed_data": final_preprocessed}
    

