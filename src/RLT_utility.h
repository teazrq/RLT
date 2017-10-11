//  **********************************************************************
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
//
//    Written by:
//
//      Ruoqing Zhu, Ph.D.
//      Assistant Professor,
//      Department of Statistics,
//      University of Illinois Urbana-Champaign
//
//      725 S. Wright St, 116 D,
//      Champaign, IL 61820
//
//      email:  teazrq@gmail.com
//      URL:    https://sites.google.com/site/teazrq/
//
//  **********************************************************************

#ifndef RLT_utility
#define RLT_utility

// parameter structure

typedef struct _PARAMETERS
{
	int model;
	int summary;
	int useCores;
	int data_n;
	int dataX_p;

	int ntrees;
	int mtry;
	int nmin;
	int split_gen;
	int nspliteach;
	int select_method;

	int nclass;
	int replacement;
	int npermute;
	int importance;
	int track_obs;

	int reinforcement;
	int muting;
	int protectVar;
	int combsplit;
	int random_select;

	int ntrees_embed;
	int nmin_embed;
	int split_gen_embed;
	int nspliteach_embed;
	int naive_embed;

	int use_sub_weight;
	int use_var_weight;

	double resample_prob;
	double muting_percent;
	double combsplit_th;
	double resample_prob_embed;
	double mtry_embed;
} PARAMETERS;

typedef struct _SplitRule
{
	int NCombinations;
	int OneSplitVariable;
	int* SplitVariables;
	double* Loadings;
	double SplitValue;

	int MutingUpdate;
	int newp;
	int* NewUseVariable;

	int ProtectUpdate;
	int* NewProtectVariable;
} SplitRule;

// structure for regression observation

typedef struct _struct_xy
{
	double x;
	double y;
} struct_xy;

typedef struct _struct_xyw
{
	double x;
	double y;
	double w;
} struct_xyw;

// structure for classification observation

typedef struct _struct_xc
{
	double x;
	int y;
} struct_xc;

typedef struct _struct_xcw
{
	double x;
	int y;
	double w;
} struct_xcw;

// structure for survival observation

typedef struct _struct_xyc
{
	double x;
	int y;
	int c;
} struct_xyc;

typedef struct _struct_xycw
{
	double x;
	int y;
	int c;
	double w;
} struct_xycw;

typedef struct _struct_idd
{
	int i;
	double d1;
	double d2;
} struct_idd;

typedef struct _struct_cat_cla
{
	int cat;
	double* cla;
	double wsum;
} struct_cat_cla;

typedef struct _struct_ifc
{
	int cat;
	int f;
	int c;
	int* flist;
	int* clist;
} struct_ifc;


void copyParameters(PARAMETERS* myPara, SEXP parameters_int, SEXP parameters_double);
void printParameters(PARAMETERS* myPara);

// random generating

int random_in_range( int min, int max);
double* runif_d_vec(double* x, int n, double min, double max);
int sample(double* x, int n);

// other functions

int imin(int a, int b);
int imax(int a, int b);
void standardize(double* x, int n);
int CheckIdentical_d(double* y, int* obs, int n);
int CheckIdentical_i(int* y, int* obs, int n);
void get_max_min(double *xmin, double *xmax, double *x, int *useObs, int n);

void print_i_vec(int *x, int n);
void print_i_vec_t(int *x, int n);
void print_d_vec(double *x, int n);
void print_i_d_vec(int *x, double *y,  int n);
void print_d_d_vec(double* x, double *y, int n);
void print_i_d_d_vec(int* x, double *y, double *z, int n);
void print_xy_struct(struct_xy* a, int n);
void print_xyw_struct(struct_xyw* a, int n);
void print_idd_struct(struct_idd* x, int n);
void print_d_mat(double** x, int d1, int d2);
void print_d_mat_t(double** x, int d1, int d2);

// sort
int sgn_rand(double x);

int compare_d (const void * a, const void * b);
int compare_struct_xy (const void *a, const void* b);
int compare_struct_xyw (const void *a, const void* b);
int compare_struct_xc (const void *a, const void* b);
int compare_struct_xcw (const void *a, const void* b);
int compare_struct_xyc (const void *a, const void* b);
int compare_struct_xycw (const void *a, const void* b);
int compare_struct_idd (const void *a, const void* b);
int compare_struct_idd_rev (const void *a, const void* b);

int compare_struct_cat_cla_two (const void *a, const void* b);
int compare_struct_cat_cla_two_rev (const void *a, const void* b);

// swap
void swap_idd(struct_idd* a, struct_idd* b);
void swap_cat_cla(struct_cat_cla* a, struct_cat_cla* b);
void swap_ifc(struct_ifc* a, struct_ifc* b);

// permutation

void permute(int* x, int n);

// categorical variable related

double pack(const int nBits, const int *bits);
void unpack(const double pack, const int nBits, int *bits);
int unpack_goright(const double pack, const int cat, const int nBits);

#endif
