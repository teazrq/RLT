.onAttach <- function(libname, pkgname){
  RFver <- read.dcf(file=system.file("DESCRIPTION", package=pkgname),
                    fields="Version")
  packageStartupMessage(paste("Reinforcement Learning Trees (", pkgname, ") v", RFver, sep = ""))
}
