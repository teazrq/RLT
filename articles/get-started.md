# Get Started with RLT

## Install the RLT Package

### Released Version

Install from CRAN when the released version is available:

``` r

install.packages("RLT")
```

### Development Version

Install the development version from GitHub:

``` r

# install.packages("remotes")
remotes::install_github("teazrq/RLT")
```

### For Windows Users

Source installation on Windows requires Rtools. Install the matching
version from [the Rtools
page](https://cran.r-project.org/bin/windows/Rtools/), then install RLT
from CRAN or GitHub.

### For macOS Users

When installing RLT on macOS, the process can be slightly trickier. It
mainly consists of two steps:

#### Step 1: Install Compilers

- Follow [this guide](https://mac.r-project.org/tools/) to install the
  GNU Fortran compiler:
  [gfortran-12.2-universal.pkg](https://mac.r-project.org/tools/gfortran-12.2-universal.pkg)
- If you do not already have Xcode (most systems come with Xcode
  installed already), you can use this line to install it:

``` sh
sudo xcode-select --install
```

#### Step 2: Set Makevars to Point to the Compiler

Create the Makevars file using:

``` sh
mkdir ~/.R
touch ~/.R/Makevars
```

Check your folder of gfortran. The compiler will be installed to
`/opt/gfortran` folder as default.

Add these lines to point to the correct folder of the compiler. You may
use `open -a TextEdit ~/.R/Makevars` to open a text editor and type in
these lines:

    FC = /opt/gfortran/bin/gfortran
    F77 = /opt/gfortran/bin/gfortran
    FLIBS = -L/opt/gfortran/lib

After completing these steps, you should be able to install the RLT
package using `remotes::install_github("teazrq/RLT")` in RStudio. The
compilation may require a few minutes.

### OpenMP in macOS

The previous steps would not activate OpenMP parallel computing. To
enable OpenMP while compiling the RLT package, follow these steps:

#### Step 1: Download and Install LLVM

For example, to install the 14.0.6 version (check the ones that
compatible with your Xcode version), use:

``` sh
curl -O https://mac.r-project.org/openmp/openmp-14.0.6-darwin20-Release.tar.gz
sudo tar fvxz openmp-14.0.6-darwin20-Release.tar.gz -C /
```

You should then see the following message:

    usr/local/lib/libomp.dylib
    usr/local/include/ompt.h
    usr/local/include/omp.h
    usr/local/include/omp-tools.h

#### Step 2: Add Flags into Makevars

The procedure is the same as we explained previously. You should add
these two lines into your Makevars file:

    CPPFLAGS += -Xclang -fopenmp
    LDFLAGS += -lomp

After these two steps, you can then use
`remotes::install_github("teazrq/RLT", force = TRUE)` to re-install the
RLT package.

## Load the RLT Package

``` r

library(RLT)
```

## Next steps

See the
[Regression](https://teazrq.github.io/RLT/articles/regression-tutorial.md),
[Classification](https://teazrq.github.io/RLT/articles/classification-tutorial.md),
and
[Survival](https://teazrq.github.io/RLT/articles/survival-tutorial.md)
tutorials to get started with modeling.
