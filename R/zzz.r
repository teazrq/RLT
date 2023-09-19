.onAttach <- function(libname, pkgname){
  pkgver <- read.dcf(file=system.file("DESCRIPTION", package=pkgname),
                    fields="Version")
  packageStartupMessage(paste("RLT and Random Forests v", pkgver, "\n",
                              "pre-release at github.com/teazrq/RLT", sep = ""))
}