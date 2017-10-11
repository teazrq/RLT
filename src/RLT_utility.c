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

# include <omp.h>
# include <string.h>
# include <Rconfig.h>
# include <Rdefines.h>
# include <Rembedded.h>
# include <R.h>
# include <Rinternals.h>
# include <Rmath.h>
# include <Rversion.h>

# include <S.h>
# include <time.h>
# include <math.h>
# include <stdio.h>
# include <stdlib.h>
# include <pthread.h>

// tool functions
# include "RLT_utility.h"


// parameters structure and print function//

void copyParameters(PARAMETERS* myPara, SEXP parameters_int, SEXP parameters_double)
{
	myPara->summary = INTEGER(parameters_int)[0];
	myPara->useCores = INTEGER(parameters_int)[1];

	myPara->ntrees = INTEGER(parameters_int)[2];
	myPara->mtry = INTEGER(parameters_int)[3];
	myPara->nmin = INTEGER(parameters_int)[4];
	myPara->split_gen = INTEGER(parameters_int)[5];
	myPara->nspliteach = INTEGER(parameters_int)[6];
	myPara->select_method = INTEGER(parameters_int)[7];

	myPara->nclass = INTEGER(parameters_int)[8];
	myPara->replacement = INTEGER(parameters_int)[9];
	myPara->npermute = INTEGER(parameters_int)[10];

	myPara->reinforcement = INTEGER(parameters_int)[11];
	myPara->muting = INTEGER(parameters_int)[12];
	myPara->protectVar = INTEGER(parameters_int)[13];
	myPara->combsplit = INTEGER(parameters_int)[14];

	myPara->ntrees_embed = INTEGER(parameters_int)[15];
	myPara->nmin_embed = INTEGER(parameters_int)[16];
	myPara->split_gen_embed = INTEGER(parameters_int)[17];
	myPara->nspliteach_embed = INTEGER(parameters_int)[18];
	myPara->naive_embed = INTEGER(parameters_int)[19];

	myPara->importance = INTEGER(parameters_int)[20];
	myPara->use_sub_weight = INTEGER(parameters_int)[21];
	myPara->use_var_weight = INTEGER(parameters_int)[22];
	myPara->track_obs = INTEGER(parameters_int)[23];

	myPara->random_select = INTEGER(parameters_int)[24];

	myPara->resample_prob = REAL(parameters_double)[0];
	myPara->muting_percent = REAL(parameters_double)[1];
	myPara->combsplit_th = REAL(parameters_double)[2];
	myPara->resample_prob_embed = REAL(parameters_double)[3];
	myPara->mtry_embed = REAL(parameters_double)[4];
}

void printParameters(PARAMETERS* myPara)
{
	Rprintf("RLT all tuning parameters detail: ---------------------------------------\n");
	Rprintf("Model                      									  = %s \n", myPara->model == 1 ? "Regression" : myPara->model == 2 ? "Classification" : "Survival");
	Rprintf("Use CPU cores:                                          useCores = %i \n", myPara->useCores);
	Rprintf("Data number of observations:                                   n = %i \n", myPara->data_n);
	Rprintf("Data number of features:                                       p = %i \n", myPara->dataX_p);
	Rprintf("Number of trees:                                          ntrees = %i \n", myPara->ntrees);
	Rprintf("Number of variables try at each split:                      mtry = %i \n", myPara->mtry);
	Rprintf("Minimum terminal node size:                                 nmin = %i \n", myPara->nmin);
	// Rprintf("Selection method:                                  select_method = %i \n", myPara->select_method);
	Rprintf("Splitting point generating method:                     split_gen = %s \n", myPara->split_gen == 1 ? "Random" : myPara->split_gen == 2 ? "Uniform" : myPara->split_gen == 3 ? "Rank" : "Best");
	if (myPara->split_gen != 4)
	Rprintf("Number of random splits:                              nspliteach = %i \n", myPara->nspliteach);

	Rprintf("Sample with replacement:                             replacement = %s \n", myPara->replacement ? "Yes" : "No");
	Rprintf("Re-sampling proportion:                            resample_prob = %2.1f%% \n", myPara->resample_prob*100);
	Rprintf("Number of OOB data impute:                              npermute = %i \n", myPara->npermute);
	Rprintf("Subject weights used:                             use_sub_weight = %s \n", myPara->use_sub_weight ? "Yes" : "No");
	Rprintf("Variable weights used:                            use_sub_weight = %s \n", myPara->use_var_weight ? "Yes" : "No");
	Rprintf("Track each observations:                               track_obs = %s \n", myPara->track_obs ? "Yes" : "No");
	Rprintf("Use reinforcement learning:                        reinforcement = %s \n", myPara->reinforcement ? "Yes" : "No");

	if (myPara->reinforcement == 1)
	{
		Rprintf("----naive embed model mode:                          naive_embed = %s \n", myPara->naive_embed ? "Yes" : "No");
		Rprintf("----Use variable muting:                                  muting = %s \n", myPara->muting == -1 ? "By percent" : myPara->muting == 0 ? "No" : "By count");

		if (myPara->muting == -1)
			Rprintf("----Muting speed (as percentage):                 muting_percent = %2.1f%% \n", myPara->muting_percent*100);

		Rprintf("----Number of protected variables:                    protectVar = %i \n", myPara->protectVar);
		Rprintf("----Use linear combination split:                      combsplit = %i \n", myPara->combsplit);
		Rprintf("----Linear combination threshold:                   combsplit_th = %.2f \n", myPara->combsplit_th);
		Rprintf("----Number of embedded trees:                       ntrees_embed = %i \n", myPara->ntrees_embed);
		Rprintf("----Embedded trees minimum terminal node size:        nmin_embed = %i \n", myPara->nmin_embed);
		Rprintf("----Embedded trees re-sampling probability:  resample_prob_embed = %2.1f%% \n", myPara->resample_prob_embed*100);
		Rprintf("----Embedded trees split_gen:                    split_gen_embed = %s \n", myPara->split_gen_embed == 1 ?  "Random" : myPara->split_gen_embed == 2 ? "Uniform" : myPara->split_gen_embed == 3 ? "Rank" : "Best");
		Rprintf("----Embedded trees nspliteach:                  nspliteach_embed = %i \n", myPara->nspliteach_embed);
		Rprintf("----Embedded trees mtry (number or proportion):       mtry_embed = %2.1f%% \n", myPara->mtry_embed*100);
	}
	Rprintf("-------------------------------------------------------------------------\n");
}

// random number generating ///

double* runif_d_vec(double* x, int n, double min, double max)
{
	int i;

	for (i=0; i<n; i++)
		x[i] = runif(min, max);
	return x;
}

int random_in_range(int min, int max)
{
  if (min == max)
    return min;

  double u;
  do {u = runif((double) min, (double) max);} while (u <= min || u >= max);
  return (int) u; // generates integers from min to max-1
}

int sample(double* x, int n)
{
	double a = unif_rand();
	int i;

	for (i = 0; i< n; i++)
	{
		a = a - x[i];
		if (a <= 0)
			return i;
	}

	Rprintf("The weight vector does not add up to 1... \n");
	return(random_in_range(0, n));
}

// other functions

int imin(int a, int b)
{
	if (a <= b)
		return a;
	return b;
}

int imax(int a, int b)
{
	if (a>=b)
		return a;
	return b;
}

void standardize(double* x, int n)
{
	double sum = 0;
	int i;

	for (i=0; i<n; i++)
		sum += x[i];

	for (i=0; i<n; i++)
		x[i] = x[i]/sum;
}

int CheckIdentical_d(double* y, int* obs, int n)
{
	double y0 = y[obs[0]];
	for (int i =1; i < n; i++)
		if (y[obs[i]] != y0)
			return(0);

	return(1);
}

int CheckIdentical_i(int* y, int* obs, int n)
{
	int y0 = y[obs[0]];
	for (int i =1; i < n; i++)
		if (y[obs[i]] != y0)
			return(0);

	return(1);
}

void get_max_min(double *xmin, double *xmax, double *x, int *useObs, int n)
{
	int i;
	double temp;

	for (i=0; i<n; i++)
	{
		temp = x[useObs[i]];
		if (temp < *xmin)
		{
			*xmin = temp;
		}else if (temp > *xmax)
		{
			*xmax = temp;
		}
	}
}

// print vectors

void print_i_vec(int *x, int n)
{
	int i;
	Rprintf("\n start to print int vector \n");
	for (i=0; i<n; i++)
		Rprintf("%i \n", x[i]);
	Rprintf("stop printing \n\n");
}

void print_i_vec_t(int *x, int n)
{
	int i;
	Rprintf("\n start to print int vector \n");
	for (i=0; i<n; i++)
		Rprintf("%i ", x[i]);
	Rprintf("\n stop printing \n\n");
}

void print_d_vec(double *x, int n)
{
	int i;
	Rprintf("\n start to print int vector \n");
	for (i=0; i<n; i++)
		Rprintf("%f \n", x[i]);
	Rprintf("stop printing \n\n");
}

void print_d_vec_t(double *x, int n)
{
	Rprintf("\n start to print double vector \n");
	for (int i=0; i<n; i++)
		Rprintf("%5.5f  ", x[i]);
	Rprintf("stop printing \n\n");
}

void print_i_d_vec(int *x, double *y,  int n)
{
	int i;
	Rprintf("\n start to print int double vector \n");
	for (i=0; i<n; i++)
		Rprintf("%i  %f \n", x[i], y[i]);
	Rprintf("stop printing \n\n");
}

void print_d_d_vec(double* x, double *y, int n)
{
	int i;
	Rprintf("\n start to print double double vectors \n");
	for (i=0; i<n; i++)
		Rprintf("%f  %f \n", x[i], y[i]);
	Rprintf("stop printing \n\n");
}

void print_i_d_d_vec(int* x, double *y, double *z, int n)
{
	int i;
	Rprintf("\n start to print int double double vectors \n");
	for (i=0; i<n; i++)
		Rprintf("%i  %f  %f \n", x[i], y[i], z[i]);
	Rprintf("stop printing \n\n");
}

void print_xy_struct(struct_xy* a, int n)
{
	int i;
	Rprintf("\n start to print xy structure \n");
	for (i=0; i<n; i++)
		Rprintf("%f  %f \n", a[i].x, a[i].y);
	Rprintf("stop printing \n\n");
}

void print_xyw_struct(struct_xyw* a, int n)
{
	int i;
	Rprintf("\n start to print xyw structure \n");
	for (i=0; i<n; i++)
		Rprintf("%f  %f  %f \n", a[i].x, a[i].y, a[i].w);
	Rprintf("stop printing \n\n");
}

void print_idd_struct(struct_idd* x, int n)
{
	int i;
	Rprintf("\n start to print int double double structure \n");
	for (i=0; i<n; i++)
		Rprintf("%i  %f  %f \n", x[i].i, x[i].d1, x[i].d2);
	Rprintf("stop printing \n\n");
}

void print_d_mat(double** x, int d1, int d2)
{
	int i, j;
	Rprintf("\n start to print double matrix \n");

	for (i=0; i<d1; i++)
	{
		for (j=0; j<d2; j++)
			Rprintf("%05.1f ", x[i][j]);
		Rprintf("\n");
	}
	Rprintf("stop printing \n\n");
}

void print_d_mat_t(double** x, int d1, int d2)
{
	int i, j;
	Rprintf("\n start to print double matrix \n");

	for (i=0; i<d2; i++)
	{
		for (j=0; j<d1; j++)
			Rprintf("%05.1f ", x[j][i]);
		Rprintf("\n");
	}
	Rprintf("stop printing \n\n");
}

// sorting related

int sgn_rand(double x)
{
	if(x > 0) return 1;
	if(x < 0) return -1;

	if (unif_rand()>0.5)
		return 1;
	else
		return -1;
}


int compare_d (const void *a, const void* b)
{
	double diff = *(double*)a - *(double*)b;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_xy (const void *a, const void* b)
{
	double diff = ((struct_xy*)a)->x - ((struct_xy*)b)->x;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_xyw (const void *a, const void* b)
{
	double diff = ((struct_xyw*)a)->x - ((struct_xyw*)b)->x;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_xc (const void *a, const void* b)
{
	double diff = ((struct_xc*)a)->x - ((struct_xc*)b)->x;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_xcw (const void *a, const void* b)
{
	double diff = ((struct_xcw*)a)->x - ((struct_xcw*)b)->x;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_xyc (const void *a, const void* b)
{
	double diff = ((struct_xyc*)a)->x - ((struct_xyc*)b)->x;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_xycw (const void *a, const void* b)
{
	double diff = ((struct_xycw*)a)->x - ((struct_xycw*)b)->x;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_idd (const void *a, const void* b)
{
	double diff = ((struct_idd*)a)->d2 - ((struct_idd*)b)->d2;
	if (diff > 0) return 1;
	if (diff < 0) return -1;
	return 0;
}

int compare_struct_idd_rev (const void *a, const void* b)
{
	double diff = ((struct_idd*)a)->d2 - ((struct_idd*)b)->d2;
	if (diff > 0) return -1;
	if (diff < 0) return 1;
	return 0;
}

int compare_struct_cat_cla_two (const void *a, const void* b)
{
	double diff = ((struct_cat_cla*)a)->cla[0]/(((struct_cat_cla*)a)->cla[0] + ((struct_cat_cla*)a)->cla[1])
				  - ((struct_cat_cla*)b)->cla[0]/(((struct_cat_cla*)b)->cla[0] + ((struct_cat_cla*)b)->cla[1]);

	if (diff > 0) return 1;
	if (diff < 0) return -1;
	if (unif_rand()>0.5) return 1; else return -1;
}

int compare_struct_cat_cla_two_rev (const void *a, const void* b)
{
	double diff = ((struct_cat_cla*)a)->cla[0]/(((struct_cat_cla*)a)->cla[0] + ((struct_cat_cla*)a)->cla[1])
				  - ((struct_cat_cla*)b)->cla[0]/(((struct_cat_cla*)b)->cla[0] + ((struct_cat_cla*)b)->cla[1]);

	if (diff > 0) return -1;
	if (diff < 0) return 1;
	if (unif_rand()>0.5) return 1; else return -1;
}

// swap

void swap_idd(struct_idd *a, struct_idd *b)
{
	struct_idd temp = *a;
	*a = *b;
	*b = temp;
}

void swap_cat_cla(struct_cat_cla* a, struct_cat_cla* b)
{
	struct_cat_cla temp = *a;
	*a = *b;
	*b = temp;
}

void swap_ifc(struct_ifc* a, struct_ifc* b)
{
	struct_ifc temp = *a;
	*a = *b;
	*b = temp;
}

// permutation

void permute(int* x, int n)
{
	int i;
	int j;
	int temp;

	for (i = 0; i<n-1; i++)
	{
		j = random_in_range(i, n);
		temp = x[i];
		x[i] = x[j];
		x[j] = temp;
	}
}
// categorical variable related

double pack(const int nBits, const int *bits) // from Andy's rf package
{
    int i;
	double value = bits[nBits - 1];

    for (i = nBits - 2; i >= 0; i--)
		value = 2.0*value + bits[i];

    return(value);
}

void unpack(const double pack, const int nBits, int *bits) // from Andy's rf package
{
    int i;
    double x = pack;
    for (i = 0; i < nBits; ++i)
	{
    	bits[i] = ((unsigned long) x & 1) ? 1 : 0;
    	x /= 2;
    }
}

int unpack_goright(const double pack, const int cat, const int nBits)
{
    int i;
    double x = pack;

    for (i = 0; i < cat; i++) x /= 2;

	return(((unsigned long) x & 1) ? 1 : 0);
}













