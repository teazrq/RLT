.onAttach <- function(libname, pkgname){
  pkgver <- read.dcf(file=system.file("DESCRIPTION", package=pkgname),
                    fields="Version")
  packageStartupMessage(paste("Random forests and RLT v", pkgver, "\n",
                              "prerelease at https://github.com/teazrq/RLT", sep = ""))
}