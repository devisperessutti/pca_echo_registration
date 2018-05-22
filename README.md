# Python implementation of PCA-based registration of multi-temporal echocardiography sequences.

This code implements the algorithm presented in [1].

## Implemented registrations

Registration of echocardiography sequences:
 * registration using the PCA-based similarity measure;
 * registration using the SSD similarity measure;
 * registration using the phase-based similarity measure.

PCA class:
 * implementation of unsupervised/supervised PCA in standard and dual formulation.  

## Required modules
Required modules:
 * numpy
 * matplotlib
 * SimpleITK (v0.8).

## References

[1] Peressutti D. _et al._ "Registration of multiview echocardiography sequences using a subspace error metric".
_IEEE Transactions on Biomedical Engineering_, 64, 352-361, 2017 
