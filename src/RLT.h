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

// RLT - Main Header File (Reorganized Structure)
// This file provides the single entry point for all RLT includes

#ifndef RLT_H
#define RLT_H

// ============================================================================
// Core Infrastructure (always required)
// ============================================================================
#include "include/core/Utility.h"
#include "include/core/Tree_Definition.h"
#include "include/core/Tree_Function.h"
#include "include/core/Stat_Function.h"
#include "include/core/Forest_Build_Guards.h"
#include "include/core/Variance_IJ_Jack.h"

// ============================================================================
// Regression Module
// ============================================================================
#include "include/regression/Reg_Uni_Definition.h"
#include "include/regression/Reg_Uni_Function.h"

// ============================================================================
// Classification Module
// ============================================================================
#include "include/classification/Cla_Uni_Definition.h"
#include "include/classification/Cla_Uni_Function.h"

// ============================================================================// Survival Module
// ============================================================================
#include "include/survival/Surv_Uni_Definition.h"
#include "include/survival/Surv_Uni_Function.h"

// ============================================================================
// Quantile Module
// ============================================================================
#include "include/quantile/Quan_Uni_Definition.h"
#include "include/quantile/Quan_Uni_Function.h"

#endif // RLT_H
