# RLT 4.2.6

# RLT 4.2.5

* Completed linear combination splits and reinforcement learning mode for regression. 
* Single variable reinforcement learning mode for classification is also done. 

# RLT 4.1.3

* Various updates including reproducibility, new models, and speed improvements. 
* Embedded model (in RLT) temporally removed. Will be added to next version. 

# RLT 4.0.0

* Updated the entire package to Rcpp and Rcpparmadillo.

# RLT 3.2.3

* Removed S.h header file.
* Changes header files to avoid conflict between Rinternals.h and clang 13.0.0.

# RLT 3.2.2

* Changed the variable weights mechanism for classification setting.

# RLT 3.2.1

* Fixed compiling issues for solaris.

# RLT 3.2.0

* Fixed compiling issues for omp.h.

# RLT 3.1.1

* Fixed a small bug in the survival model, where the "rank" splitting rule may lead to a crash.

# RLT 3.1.0

* This is the first release of this package on CRAN. Previous versions (<= 3.0.0) were released on the author's personal website. 
* There are some minor changes to the parameter names compared with previous versions.