import warnings
import os
import numpy as np
from gapp import gp, dgp, covariance
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.integrate import quad

def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Hubble parameter for a flat LCDM model
def hz_model(z, omegaM, h):
    return h*np.sqrt( omegaM*(1.+z)**3. + (1.-omegaM) )

# comoving radial distance integrand for a flat LCDM MODEL
def integrand(z, omegaM, h):
    return 1./np.sqrt( omegaM*(1.+z)**3. + (1.-omegaM) )

# comoving radial distance calculation assuming the above integrand
def dz_model(z, omegaM, h):
    aux = quad(integrand, 0., z, args=(omegaM, h))
    aux = aux[0]
    c = 2.998e5
    DH = c/h
    return DH*aux

# deceleration parameter for a flat LCDM model
def qz_model(z, omegaM):
    return (1./2.)*((3.*omegaM*(1.+z)**3.)/(omegaM*(1.+z)**3. + 1.-omegaM)) - 1.

# Hubble parameter derivative for a flat LCDM model
def dhz_model(z, omegaM, h):
    return ( (3./2.)*omegaM*(1.+z)**2.*h )/np.sqrt( omegaM*(1.+z)**3. + 1.-omegaM )

if __name__ == "__main__":

    local_path = os.getcwd()

    # ---------------- loading data (SDSS and/or DESI BAO)
    file_name = 'desi_obs'
    # file_name = 'sdssbao_obs'
    # file_name = 'desi_sdssbao_obs_C1'
    # file_name = 'desi_sdssbao_obs_C2'
    (z, dhrdz, errdhrdz) = np.loadtxt('dhrdz_'+file_name+'.txt', unpack='True')
    
    # ---------------- loading data (CC+SDSS (old))
    # file_name = 'hz_cc_sdss'
    # (z, Hz, errHz, hid) = np.loadtxt(file_name+'.txt', unpack='True')

    # rd prior
    rdprior = 'p18'
    # rdprior= 'lowz'

    # ---------------- computing hz from dhrdz, propagating the error on rd*h as reported by arXiv:1607.05297 for h(z) reconstruction, assuming sound horizon rd*h from
    # (dhrdz) = c/(rd*Hz) => Hz = c/(rd*dhrdz), thus hz = Hz/H0 = c/(rd*H0*dhrdz) = c/(100*rdh*dhrdz), c = speed of light (km/s), rd = sound horizon scale (Mpc)
    # c = 2.998e5
    # rdh = 101.0
    # errrdh = 2.3
    # hz = c/(100.*rdh*dhrdz)
    # errhz = np.sqrt( (c/(100.*rdh**2.*dhrdz))**2.*errrdh**2. + (c/(100.*rdh*(dhrdz**2)))**2.*errdhrdz**2. )

    # ---------------- computing Hz from dhrdz, propagating the error on rd as reported by Planck18 (page 28 in arXiv:1807.06209)
    # (dhrdz) = c/(rd*Hz) => Hz = c/(rd*dhrdz), c = speed of light (km/s), rd = sound horizon scale (Mpc)
    if rdprior == 'p18':
        c = 2.998e5
        rd = 147.05
        errrd = 0.3
        Hz = c/(rd*dhrdz)
        errHz = np.sqrt((c/(rd**2.*dhrdz))**2.*errrd**2. + +
                        (c/(rd*(dhrdz**2)))**2.*errdhrdz**2.)
        om = 0.315
        ox = 1-om
        H0 = 67.4
        errH0 = 0.5
        w0 = -1.
        wa = 0.
        # print z, Hz, errHz

    # ---------------- same as above, but assuming rd from R18 H0 and Pantheon+BAO reported by Planck18 (page 28 in arXiv:1807.06209)
    # (dhrdz) = c/(rd*Hz) => Hz = c/(rd*dhrdz), c = speed of light (km/s), rd = sound horizon scale (Mpc)
    if rdprior == 'lowz':
        c = 2.998e5
        rd = 136.4
        errrd = 3.5
        Hz = c/(rd*dhrdz)
        errHz = np.sqrt((c/(rd**2.*dhrdz))**2.*errrd**2. + +
                        (c/(rd*(dhrdz**2)))**2.*errdhrdz**2.)
        om = 0.334
        ox = 1-om
        H0 = 73.3
        errH0 = 1.04
        w0 = -1.
        wa = 0.

    # ===================== GAUSSIAN PROCESS RECONSTRUCTION

    # --------- defining the redshift range and the number of bins of the GP reconstruction
    zmin = 0.0001
    zmax = 2.50
    nbins = 250

    # kernel to be used
    kernel = 'sqexp'
    # kernel = 'mat72'

    #  --------- performing the reconstruction using the dgp module from GaPP in the [zmin,zmax] range:
    if kernel == 'mat72':
        g = dgp.DGaussianProcess(z, Hz, errHz, covfunction=covariance.Matern72,
                                 cXstar=(zmin, zmax, nbins))

    if kernel == 'sqexp':
        g = dgp.DGaussianProcess(z, Hz, errHz, covfunction=covariance.SquaredExponential,
                                 cXstar=(zmin, zmax, nbins))

    (hzrec, theta) = g.gp(thetatrain='True')
    (dhzrec, theta) = g.dgp(thetatrain='False')
    # (d2hzrec, theta) = g.d2gp()
    # (d3hzrec, theta) = g.d3gp()

    # calculate covariances between Hz and Hz' at points Zstar.
    fcov_hzdhz = g.f_covariances(fclist=[0, 1])
    
    # --------- debug (likelihood)
    # print g.log_likelihood()

    # --------- getting the reconstructed quantities hz, dhz, their errors and covariances
    n_start = 0
    z_rec = hzrec[n_start:, 0]
    hz_rec = hzrec[n_start:, 1]
    errhz_rec = hzrec[n_start:, 2]
    dhz_rec = dhzrec[n_start:, 1]
    errdhz_rec = dhzrec[n_start:, 2]
    errhzdhz_rec = fcov_hzdhz[n_start:, :,]

    # getting Ez = Hz/H0
    ez_rec = hz_rec/H0
    errez_rec = np.sqrt( ((errhz_rec)/(H0**2.))**2.*errH0**2. + (1./H0)**2.*errhz_rec**2. )

    # --------- computing om and its respective uncertainty via error propagation
    omz_rec = ( ez_rec**2.-1. )/( (1.+z_rec)**3. - 1.)
    erromz_rec = ( 2.*ez_rec*errez_rec )/(1.+z_rec)**3.

    # --------- computing qz and its respective uncertainty via error propagation
    qz_rec = ((1.+z_rec)*dhz_rec/hz_rec)-1.
    errqz_rec = np.sqrt((errhz_rec/hz_rec)**2. + (errdhz_rec/dhz_rec)
                        ** 2. - 2.*errhzdhz_rec[:, 1, 0]/(hz_rec*dhz_rec))*(1.+qz_rec)

    # ===================== SAVING RESULTS

    np.savetxt('rec_Hz_'+file_name+'_rd'+rdprior+'_'+kernel+'.txt', hzrec)
    np.savetxt('rec_dHz_'+file_name+'_rd'+rdprior+'_'+kernel+'.txt', dhzrec)
    np.savetxt('rec_qz_'+file_name+'_rd'+rdprior+'_'+kernel+'.txt',
                np.transpose([z_rec, qz_rec, errqz_rec]))
    np.savetxt('rec_omz_'+file_name+'_rd'+rdprior+'_'+kernel+'.txt', 
                np.transpose([z_rec,omz_rec,erromz_rec]))

    # ===================== PLOTTING

    # latex rendering text fonts
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # # ----------- (i) plotting (DH/rd) reconstruction -- OPTIONAL!!!

    # fig, ax1 = plt.subplots(figsize = (12., 9.))

    # # Define axes
    # ax1.set_xlabel(r"$z$", fontsize=27)
    # ax1.set_ylabel(r"$(D_{\rm H}/r_{\rm d})(z)$ (Mpc)", fontsize=27)
    # ax1.set_title(r"DESI", fontsize=30)
    # plt.xlim(z_rec.min(),z_rec.max())
    # for t in ax1.get_xticklabels(): t.set_fontsize(27)
    # for t in ax1.get_yticklabels(): t.set_fontsize(27)

    # plt.plot(z_rec,dhrdz_rec, '-', color='black')
    # plt.errorbar(z,dhrdz,yerr=errdhrdz, fmt='.', color='black')
    # ax1.fill_between(z_rec, dhrdz_rec+1.*errdhrdz_rec, dhrdz_rec-1.*errdhrdz_rec, facecolor='#808080', alpha=0.80, interpolate=True)
    # ax1.fill_between(z_rec, dhrdz_rec+2.*errdhrdz_rec, dhrdz_rec-2.*errdhrdz_rec, facecolor='#808080', alpha=0.50, interpolate=True)
    # ax1.fill_between(z_rec, dhrdz_rec+3.*errdhrdz_rec, dhrdz_rec-3.*errdhrdz_rec, facecolor='#808080', alpha=0.30, interpolate=True)
    # plt.legend((r"GaPP rec", "$1\sigma$", "$2\sigma$", "$3\sigma$", "Data"), fontsize='27', loc='best')
    # #plt.show()

    # #saving the plot
    # fig.savefig('rec_dhrdz_rdhv17_'+file_name+'_sqexp.png')

    # # ----------- (ii) plotting (DH/rd)' = d(DH/rd)/dz reconstruction -- OPTIONAL!!!

    # fig, ax2 = plt.subplots(figsize = (12., 9.))

    # # Define axes
    # ax2.set_xlabel(r"$z$", fontsize=27)
    # ax2.set_ylabel(r"$(D'_{\rm H}/r_{\rm d})(z)$ (Mpc)", fontsize=27)
    # ax2.set_title(r"DESI", fontsize=30)
    # plt.xlim(z_rec.min(),z_rec.max())
    # for t in ax2.get_xticklabels(): t.set_fontsize(27)
    # for t in ax2.get_yticklabels(): t.set_fontsize(27)

    # plt.plot(z_rec,ddhrdz_rec, '-', color='black')
    # ax2.fill_between(z_rec, ddhrdz_rec+1.*errddhrdz_rec, ddhrdz_rec-1.*errddhrdz_rec, facecolor='#808080', alpha=0.80, interpolate=True)
    # ax2.fill_between(z_rec, ddhrdz_rec+2.*errddhrdz_rec, ddhrdz_rec-2.*errddhrdz_rec, facecolor='#808080', alpha=0.50, interpolate=True)
    # ax2.fill_between(z_rec, ddhrdz_rec+3.*errddhrdz_rec, ddhrdz_rec-3.*errdhrdz_rec, facecolor='#808080', alpha=0.30, interpolate=True)
    # plt.legend((r"GaPP rec", "$1\sigma$", "$2\sigma$", "$3\sigma$"), fontsize='27', loc='best')
    # #plt.show()

    # #saving the plot
    # fig.savefig('rec_ddhrdz_rdhv17_'+file_name+'_sqexp.png')

    # ---------------- (iii) plotting Hz reconstruction

    fig, ax3 = plt.subplots(figsize=(12., 9.))

    # Define axes
    ax3.set_xlabel(r"$z$", fontsize=27)
    ax3.set_ylabel(
        r"$H(z)$ ($\mathrm{km} \; \mathrm{s}^{-1} \; \mathrm{Mpc}^{-1}$)", fontsize=27)
    if file_name == 'desi_obs':
        ax3.set_title(r"DESI", fontsize=30)
    if file_name == 'sdssbao_obs':
        ax3.set_title(r"SDSS", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C1':
        ax3.set_title(r"DESI+SDSS (C1)", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C2':
        ax3.set_title(r"DESI+SDSS (C2)", fontsize=30)
    if file_name == 'hz_cc_sdss':
        ax3.set_title(r"CC+SDSS (old)", fontsize=30)
    plt.xlim(z_rec.min(), 2.4)
    plt.ylim(50.,300.)
    for t in ax3.get_xticklabels():
        t.set_fontsize(27)
    for t in ax3.get_yticklabels():
        t.set_fontsize(27)

    plt.plot(z_rec, hz_rec, '-', color='black')
    plt.plot(z_rec, hz_model(z_rec,om,H0), '-.', color='blue')
    plt.errorbar(z, Hz, yerr=errHz, fmt='.', color='black')
    ax3.fill_between(z_rec, hz_rec+1.*errhz_rec, hz_rec-1.*errhz_rec,
                     facecolor='#808080', alpha=0.80, interpolate=True)
    ax3.fill_between(z_rec, hz_rec+2.*errhz_rec, hz_rec-2.*errhz_rec,
                     facecolor='#808080', alpha=0.50, interpolate=True)
    ax3.fill_between(z_rec, hz_rec+3.*errhz_rec, hz_rec-3.*errhz_rec, 
                     facecolor='#808080', alpha=0.30, interpolate=True)
    plt.legend((r"GaPP rec", "$\Lambda$CDM", "$1\sigma$", "$2\sigma$", "$3\sigma$", "data"),
               fontsize='27', loc='lower right')
    # plt.show()

    # saving the plot
    fig.savefig('rec_Hz_'+file_name+'_rd'+rdprior+'_'+kernel+'.png')

    # ---------------- (iv) plotting H'z reconstruction

    fig, ax4 = plt.subplots(figsize=(12., 9.))

    # Define axes
    ax4.set_xlabel(r"$z$", fontsize=27)
    ax4.set_ylabel(
        r"$H'(z)$ ($\mathrm{km} \; \mathrm{s}^{-1} \; \mathrm{Mpc}^{-1}$)", fontsize=27)
    if file_name == 'desi_obs':
        ax4.set_title(r"DESI", fontsize=30)
    if file_name == 'sdssbao_obs':
        ax4.set_title(r"SDSS", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C1':
        ax4.set_title(r"DESI+SDSS (C1)", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C2':
        ax4.set_title(r"DESI+SDSS (C2)", fontsize=30)
    if file_name == 'hz_cc_sdss':
        ax4.set_title(r"CC+SDSS (old)", fontsize=30)
    plt.xlim(z_rec.min(), 2.4)
    plt.ylim(-150.,150.)
    for t in ax4.get_xticklabels():
        t.set_fontsize(27)
    for t in ax4.get_yticklabels():
        t.set_fontsize(27)

    plt.plot(z_rec, dhz_rec, '-', color='black')
    plt.plot(z_rec, dhz_model(z_rec,om,H0), '-.', color='blue')
    ax4.fill_between(z_rec, dhz_rec+1.*errdhz_rec, dhz_rec-1. *
                     errdhz_rec, facecolor='#808080', alpha=0.80, interpolate=True)
    ax4.fill_between(z_rec, dhz_rec+2.*errdhz_rec, dhz_rec-2. *
                     errdhz_rec, facecolor='#808080', alpha=0.50, interpolate=True)
    ax4.fill_between(z_rec, dhz_rec+3.*errdhz_rec, dhz_rec-3. * 
                     errdhz_rec, facecolor='#808080', alpha=0.30, interpolate=True)
    plt.legend((r"GaPP rec", "$\Lambda$CDM", "$1\sigma$", "$2\sigma$", "$3\sigma$"),
               fontsize='27', loc='lower right')
    # plt.show()

    # saving the plot
    fig.savefig('rec_dHz_'+file_name+'_rd'+rdprior+'_'+kernel+'.png')

    # ---------------- (v) plotting om reconstruction

    fig, ax5 = plt.subplots(figsize = (12., 9.))

    # Define axes
    ax5.set_xlabel(r"$z$", fontsize=27)
    ax5.set_ylabel(r"$\mathcal{O}_{\rm m}(z)$", fontsize=27)
    if file_name == 'desi_obs':
        ax5.set_title(r"DESI", fontsize=30)
    if file_name == 'sdssbao_obs':
        ax5.set_title(r"SDSS", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C1':
        ax5.set_title(r"DESI+SDSS (C1)", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C2':
        ax5.set_title(r"DESI+SDSS (C2)", fontsize=30)
    if file_name == 'hz_cc_sdss':
        ax5.set_title(r"CC+SDSS (old)", fontsize=30)    
    plt.xlim(0.2,2,4)
    plt.ylim(0.,1.)
    for t in ax5.get_xticklabels(): t.set_fontsize(27)
    for t in ax5.get_yticklabels(): t.set_fontsize(27)

    if rdprior == 'p18':
        omegam = 0.315
        erromegam = 0.007
    
    if rdprior == 'lowz':
        omegam = 0.334
        erromegam = 0.018
    
    plt.plot(z_rec,omz_rec, '-', color='black')
    ax5.fill_between(z_rec, omz_rec+1.*erromz_rec, omz_rec-1.*erromz_rec, facecolor='#808080', alpha=0.80, interpolate=True)
    ax5.fill_between(z_rec, omz_rec+2.*erromz_rec, omz_rec-2.*erromz_rec, facecolor='#808080', alpha=0.50, interpolate=True)
    ax5.fill_between(z_rec, omz_rec+3.*erromz_rec, omz_rec-3.*erromz_rec, facecolor='#808080', alpha=0.30, interpolate=True)
    ax5.fill_between(z_rec, omegam+1.*erromegam, omegam-1.*erromegam, facecolor='#0000FF', alpha=0.50, interpolate=True)
    plt.legend((r"GaPP rec", "$1\sigma$", "$2\sigma$", "$3\sigma$", 
                "$\Lambda$CDM"), fontsize='27', loc='best')
    #plt.show()

    #saving the plot
    fig.savefig('rec_omz_'+file_name+'_rd'+rdprior+'_'+kernel+'.png')

    # ---------------- (vi) plotting qz reconstruction

    fig, ax6 = plt.subplots(figsize=(12., 9.))

    # Define axes
    ax6.set_xlabel(r"$z$", fontsize=27)
    ax6.set_ylabel(r"$q(z)$", fontsize=27)
    if file_name == 'desi_obs':
        ax6.set_title(r"DESI", fontsize=30)
    if file_name == 'sdssbao_obs':
        ax6.set_title(r"SDSS", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C1':
        ax6.set_title(r"DESI+SDSS (C1)", fontsize=30)
    if file_name == 'desi_sdssbao_obs_C2':
        ax6.set_title(r"DESI+SDSS (C2)", fontsize=30)
    if file_name == 'hz_cc_sdss':
        ax6.set_title(r"CC+SDSS (old)", fontsize=30)
    plt.xlim(z_rec.min(), 2.4)
    plt.ylim(-3.5,1.)
    if file_name == 'hz_cc_sdss':
        plt.ylim(-1.5,1.5)
    for t in ax6.get_xticklabels():
        t.set_fontsize(27)
    for t in ax6.get_yticklabels():
        t.set_fontsize(27)
        
    plt.axhline(0., color='black')
    plt.plot(z_rec, qz_model(z_rec,om), '-.', color='blue')
    plt.plot(z_rec, qz_rec, '--', color='black')
    ax6.fill_between(z_rec, qz_rec+1.*np.abs(errqz_rec), qz_rec-1. *
                     np.abs(errqz_rec), facecolor='#808080', alpha=0.80, interpolate=True)
    ax6.fill_between(z_rec, qz_rec+2.*np.abs(errqz_rec), qz_rec-2. *
                     np.abs(errqz_rec), facecolor='#808080', alpha=0.50, interpolate=True)
    ax6.fill_between(z_rec, qz_rec+3.*np.abs(errqz_rec), qz_rec-3. *
                     np.abs(errqz_rec), facecolor='#808080', alpha=0.30, interpolate=True)
    plt.legend((r"no acc.", "$\Lambda$CDM", "GaPP rec", "$1\sigma$",
                "$2\sigma$", "$3\sigma$"), fontsize='27', loc='lower right')
    # plt.show()

    # saving the plot
    fig.savefig('rec_qz_'+file_name+'_rd'+rdprior+'_'+kernel+'.png')