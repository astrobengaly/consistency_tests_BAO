# consistency_tests_BAO
Code developed for performing consistency tests between different BAO datasets in a model-independent way, whose results are presented in the paper "Consistency tests between SDSS and DESI BAO measurements", by Basundhara Ghosh and Carlos Bengaly [https://arxiv.org/abs/2408.0443], which has been accepted for publication in Physics of the Dark Universe [https://doi.org/10.1016/j.dark.2024.101699]. 

The code consistency_tests_bao.py performs the reconstruction of the Hubble parameter, H(z), along with its respective uncertainty and its first order derivative, H'(z), by means of a Gaussian Process reconstruction using the GaPP code [https://github.com/astrobengaly/GaPP]. We use the latest BAO DH(z)/rd measurements of the Sloan Digital Sky Survey (SDSS) and Dark Energy Spectroscopic Information (DESI), besides two different joint SDSS+DESI datasets that avoid double counting of the data points. Then, we compute the deceleration parameter, q(z), as well as the null diagnostic, Om(z), that depend on the H(z) and H'(z) reconstructions (as discussed in the paper) from those datasets. 

Some observations about the code:  

(i) We assume priors for sound horizon scale at the drag epoch (rd) values consistent with the latest Planck CMB or SH0ES SN luminosity distance measurements, when appliable -- see the Planck 2018 paper for more info [https://arxiv.org/abs/1807.06209]. Different rd prior assumptions will also lead to different priors on the Hubble Constant, H0, and the matter density parameter, omega_m. 

(ii) The code is originally hardwired for DESI BAO dataset alone. If you wish to run the code for other dataset, just comment line 42 and uncomment one of the lines just below this one, which correspond to other file_names. Same goes for the prior assumption on rd, at line 53 -- which also leads to different priors on H0 and omega_m -- and the Gaussian Process kernel assumption at line 105. 

The datasets used in this analysis are also made available in this repository. Please see also the original paper by DESI collaboration [https://arxiv.org/abs/2404.03002], and the SDSS webpage [https://www.sdss4.org/science/final-bao-and-rsd-measurements-table/], where the datasets were originally taken from. 

We please ask to cite our paper [https://arxiv.org/abs/2408.0443]; [https://doi.org/10.1016/j.dark.2024.101699], in addition to this github repository, if you use some of the material available in this repository. The same goes for the DESI and SDSS BAO papers corresponding to the available data.  

Questions, or any further enquiry, can be addressed to carlosbengaly@on.br 
