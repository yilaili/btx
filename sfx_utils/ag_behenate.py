import numpy as np
import matplotlib.pyplot as plt
import radial_utils

class AgBehenate:
    
    def __init__(self):
        self.q0 = 0.1076 # |q| of first peak in Angstrom
        self.delta_qs = np.arange(0.01,0.05,0.00005) # q-spacings to scan over, better in log space?
        
    def ideal_rings(self, qPeaks):
        """
        Scan over and score various inter-peak distances in q-space, based
        on the equidistance of peaks in q-space for silver behenate powder.

        Parameters
        ----------
        qPeaks : numpy.ndarray, 1d
            positions of powder peaks in q-space (inverse Angstrom)

        Returns
        -------
        rings : numpy.ndarray, 1d
            predicted positions of peaks based on the best fit q-spacing
        scores : numpy.ndarray, 1d
            residual associated with each q-spacing, ordered as self.delta_qs
        """
        order_max=13
        qround = np.round(qPeaks,5)
        scores = [] 
        for dq in self.delta_qs:
            order  = qround/dq
            remain = np.minimum(qround%dq, np.abs(dq-(qround%dq)))
            score  = np.mean(remain[np.where(order<order_max)])/dq # %mod helps prevent half periods from scoring well
            scores.append(score)
        deltaq_current = self.delta_qs[np.argmin(scores)] # 
        rings = np.arange(deltaq_current, deltaq_current*(order_max+1), deltaq_current)
        return rings, np.array(scores)

    def detector_distance(self, est_q0, wavelength, pixel_size):
        """
        Estimate the sample-detector distance, based on the fact that the 
        first diffraction ring for silver behenate is at 0.1076/Angstrom. 

        Parameters
        ----------
        est_q0 : float
            predicted q-position of first ring based on the best fit q-spacing in Angstrom
        wavelength : float
            beam wavelength in Angstrom
        pixel_size : float
            detector pixel size in mm
        """
        distance = est_q0*pixel_size/np.tan(2.*np.arcsin(wavelength*(self.q0/(2*np.pi))/2))
        print("Detector distance inferred from powder rings: %s mm" % (np.round(distance,2)))
        return distance
    
    def opt_distance(self, powder, est_distance, pixel_size, wavelength, plot=False):
        """
        Optimize the sample-detector distance based on the powder image.
        
        Parameters
        ----------
        powder : numpy.ndarray, 2d
            powder diffraction image, in shape of assembled detector
        est_distance : float
            estimated sample-detector distance in mm
        pixel_size : float
            detector pixel size in mm
        wavelength : float
            beam wavelength in Angstrom
        plot : bool
            if True, plot the results
        
        Returns
        -------
        opt_distance : float
            optimized sample-detector distance in mm
        """
        from scipy.signal import find_peaks
        
        # determine peaks in radial intensity profile and associated positions in q
        iprofile = radial_utils.radial_profile(powder)
        peaks_observed, properties = find_peaks(iprofile, prominence=1, distance=10)
        qprofile = radial_utils.pix2q(np.arange(iprofile.shape[0]), wavelength, est_distance, pixel_size)
        
        # optimize the detector distance based on inter-peak distances
        rings, scores = self.ideal_rings(qprofile[peaks_observed])
        peaks_predicted = radial_utils.q2pix(rings, wavelength, est_distance, pixel_size)
        opt_distance = self.detector_distance(peaks_predicted[0], wavelength, pixel_size)
        
        if plot:
            self.visualize_results(powder, center=(int(powder.shape[1]/2), int(powder.shape[0]/2)), 
                                   peaks_predicted=peaks_predicted, peaks_observed=peaks_observed,
                                   scores=scores, Dq=self.delta_qs,
                                   radialprofile=iprofile, qprofile=qprofile)
        
        return opt_distance
    
    def visualize_results(self, image, mask=None, vmax=50,
                          center=None, peaks_predicted=None, peaks_observed=None,
                          scores=None, Dq=None,
                          radialprofile=None, qprofile=None):
        """
        Visualize fit, plotting (1) the residuals for the delta q scan, (2) the
        observed and predicted peaks in the radial intensity profile, and (3) the
        powder image.
        """
        fig = plt.figure(figsize=(8,12),dpi=120)
        nrow,ncol=4,3
        irow,icol=0,0
        
        # plotting residual for each inter-peak spacing
        if scores is not None:
            ax1 = plt.subplot2grid((nrow, ncol), (irow, icol))
            plt.title('Score')
            ax1.plot(Dq,scores, '--', markersize=2, linewidth=1, color='black')
            ax1.plot(Dq[np.argmin(scores)], np.min(scores), 'o', color='black')
            ax1.text(Dq[np.argmin(scores)], 0.5*np.min(scores), r'inter-ring spacing = %s $\AA^{-1}$' % (str(np.round(Dq[np.argmin(scores)],4))))
            ax1.set_ylim(0)
            ax1.set_xlabel(r'$\Delta q$ ($\AA^{-1}$)')
            icol+=1
        
        # plotting radial profiles with peaks
        if radialprofile is not None:
            ax2 = plt.subplot2grid((nrow, ncol), (irow, icol), colspan=ncol-icol)
            plt.title('Radially averaged intensity')
            ax2.plot(qprofile, radialprofile, lw=0.5, color='black')
            ax2.plot(qprofile[peaks_observed],radialprofile[peaks_observed], 'o', color='red', markersize=2, label='peaks detected')
            ax2.plot(qprofile[np.floor(peaks_predicted[peaks_predicted<radialprofile.shape[0]]).astype(int)], 
                     radialprofile[np.floor(peaks_predicted[peaks_predicted<radialprofile.shape[0]]).astype(int)], 
                     'o', color='black', markersize=2, label='peaks predicted')
            ax2.set_xlabel(r'q ($\AA^{-1}$)')
            ax2.legend()
            irow+=1
        
        # plotting powder
        ax3 = plt.subplot2grid((nrow, ncol), (irow, 0), rowspan=nrow-irow, colspan=ncol)
        if mask is not None:
            image *= mask
        ax3.imshow(image,interpolation='none',vmin=0,vmax=vmax)
        ax3.set_title('Average Silver Behenate')
        if center is not None:
            ax3.plot(center[0],center[1],'ro')
            if peaks_predicted is not None:
                for peak in peaks_predicted:
                    circle = plt.Circle((center[0],center[1]), peak, fill=False, color='black', linestyle=':')
                    ax3.add_artist(circle)
            if peaks_observed is not None:
                for peak in peaks_observed:
                    circle = plt.Circle((center[0],center[1]), peak, fill=False, color='red', linestyle=':')
