//    ----------------------------------------------------------------
//
//    Reinforcement Learning Trees (RLT)
//
//    This program is free software; you can redistribute it and/or
//    modify it under the terms of the GNU General Public License
//    as published by the Free Software Foundation; either version 3
//    of the License, or (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public
//    License along with this program; if not, write to the Free
//    Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
//    Boston, MA  02110-1301, USA.
//
//    ----------------------------------------------------------------


//#define ARMA_USE_OPENMP
#define ARMA_NO_DEBUG
//#define RLT_DEBUG

// header files
# include <RcppArmadillo.h>
# include <Rcpp.h>

# include "Utility/Tree_Definition.h"
# include "Utility/Utility.h"
# include "Utility/Tree_Function.h"

// regression
# include "RegUni/Reg_Uni_Definition.h"
# include "RegUni/Reg_Uni_Function.h"

// survival
# include "SurvUni/Surv_Uni_Definition.h"
# include "SurvUni/Surv_Uni_Function.h"