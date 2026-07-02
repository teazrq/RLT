## R CMD check results

0 errors | 0 warnings | 3 notes

* This is an update release (v6.1.0).
* The 3 NOTEs are all environmental:
  1. "unable to verify current time" — sandbox network restriction
  2. "Non-standard file/directory found at top level: 'cran-comments.md'" — standard CRAN submission file
  3. "-mno-omit-leaf-frame-pointer" — Ubuntu 26.04 system compiler default (not from package Makevars)

## Test environments

* local: R 4.5.2, Ubuntu 26.04, x86_64
* R CMD check --as-cran: 0 errors, 0 warnings, 3 notes
* devtools::test(): 326 tests passed (0 failures, 0 warnings)

## Changes in this version

* Added OOB (out-of-bag) self-kernel to `forest.kernel()` via `oob = TRUE`.
  Computes kernel co-occurrence only from trees where both observations are OOB,
  eliminating response-contamination bias for unbiased degrees-of-freedom estimation.
* Added examples to print method man pages.
* Fixed C++ compiler warning (unused variable).
* Updated URL and RoxygenNote fields in DESCRIPTION.
