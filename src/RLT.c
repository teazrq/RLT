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

// utility and sort
# include "RLT_utility.h"

// model fitting functions
# include "RLT.h"

// #define CSTACK_DEFNS 7

SEXP RLT_regression(SEXP datasetX_R,
					SEXP datasetY_R,
					SEXP ncat_R,
					SEXP subjectweight_R,
					SEXP variableweight_R,
					SEXP parameters_int_R,
					SEXP parameters_double_R)
{
	// R_CStackLimit=(uintptr_t)-1;  // raise the stack limit of R

	// get data dimension and parameters

	PARAMETERS *myPara = malloc(sizeof(PARAMETERS));
	copyParameters(myPara, parameters_int_R, parameters_double_R);

	SEXP dataX_dim = getAttrib(datasetX_R, R_DimSymbol);
	myPara->model = 1;
	myPara->data_n = INTEGER(dataX_dim)[0];
	myPara->dataX_p = INTEGER(dataX_dim)[1];

	if (myPara->summary >= 1)
	{
		Rprintf("Reinforcement Learning Trees for regression, Version 2.0 \n");
		printParameters(myPara);
	}

	int data_n = myPara->data_n;
	int dataX_p = myPara->dataX_p;
	int i;
	int j;
	int nt = 0;

	int ntrees = myPara->ntrees;
	int combsplit = myPara->combsplit;
	int importance = myPara->importance;
	int track_obs = myPara->track_obs;

	// get data
	double **dataX_matrix = (double **) malloc(dataX_p * sizeof(double *));
	for (j = 0; j < dataX_p; j++)
		dataX_matrix[j] = &REAL(datasetX_R)[j*data_n];

	double *dataY_vector = REAL(datasetY_R);
	int *ncat = INTEGER(ncat_R);

	// copy subjectweight
	double *subjectweight = REAL(subjectweight_R);
	standardize(subjectweight, data_n);	// this could change the input data due to precision loss...

	double *variableweight = REAL(variableweight_R);
	standardize(variableweight, dataX_p);	// this could change the input data due to precision loss...

	// copy partialdata
	int *partialdata = (int *) malloc (data_n * sizeof(int));
	if (partialdata == NULL) error("Unable to malloc for observation index");
	for (i = 0; i < data_n; i++)
		partialdata[i] = i;

	// variables can be used
	int *usevariable = (int *) malloc (dataX_p * sizeof(int));
	if (usevariable == NULL) error("Unable to malloc for variable index");
	for (j = 0; j < dataX_p; j++)
		usevariable[j] = j;

	// protected variables
	int *protectvariable = (int *) malloc (dataX_p * sizeof(int));
	if (protectvariable == NULL) error("Unable to malloc for variable index");
	for (j = 0; j < dataX_p; j++)
		protectvariable[j] = 0;

	// get tree matrix
	int TreeMaxLength = 1 + 2*data_n; // 3*imax((int) data_n/nmin, 1) + 1 + 20;  // if require minimum sample size
	int TreeWidth = 8 + 2*combsplit;

	// create matrices for fitted trees
	double ***tree_matrix = (double ***) malloc(ntrees * sizeof(double **));
	if (tree_matrix == NULL) error("Unable to malloc for tree matrix");

	for (nt=0; nt<ntrees; nt++)
	{
		tree_matrix[nt] = (double **) malloc(TreeMaxLength * sizeof(double*));
		if (tree_matrix[nt] == NULL) error("Unable to malloc for tree matrix");

		for (i = 0; i < TreeMaxLength; i++)
			tree_matrix[nt][i] = NULL; // if this is NULL, then is node is not used yet
	}

	// variable importance for each tree
	double **AllError = (double **) malloc(ntrees * sizeof(double *));
	if (AllError == NULL) error("Unable to calloc for mse recording");

	double *VarImp = calloc(dataX_p, sizeof(double));
	if (VarImp == NULL) error("Unable to calloc for variable importance");

	if (importance)
		for (nt=0; nt<ntrees; nt++)
		{
			AllError[nt] = (double *) calloc((dataX_p+1), sizeof(double));
			if (AllError[nt] == NULL) error("Unable to calloc for mse recording");
		}

	// observation registration
	int **ObsTrack = (int **) malloc(ntrees * sizeof(int *));
	if (track_obs)
		for (nt = 0; nt<ntrees; nt++)
		{
			ObsTrack[nt] = (int *) calloc(data_n, sizeof(int));
			if (ObsTrack[nt] == NULL) error("Unable to calloc for observation track");
		}
	////////////////////////////////////////////////////////////////////////////
	///////////     Start to fit RLT model here     ////////////////////////////
	////////////////////////////////////////////////////////////////////////////

	if (myPara->summary >= 2)
		Rprintf("--------- start to fit RLT trees ------------ \n");

/*  FILE * Output;
	Output = fopen("error.txt", "w+");
	fprintf(Output, " --- Start fitting trees --- \n");
	fclose(Output);  */

  GetRNGstate();
	Fit_Trees_regression(dataX_matrix, dataY_vector, tree_matrix, AllError, VarImp, ObsTrack, myPara, ncat, subjectweight, variableweight, partialdata, usevariable, protectvariable, data_n, dataX_p);
	PutRNGstate();

/* 	Output = fopen("error.txt", "a");
	fprintf(Output, " --- end fitting trees --- \n");
	fclose(Output);	 */

	// free some memory that is not used in the output
	free(dataX_matrix);
	free(partialdata);
	free(usevariable);
	free(protectvariable);
	free(myPara);

	// converting tree_matrix into R output

	SEXP FittedTrees_R;
	SEXP FittedOneTree;

	PROTECT(FittedTrees_R = allocVector(VECSXP, ntrees));

	SEXP set_name;  // both column and row names
	SEXP set_name_r; // just row names

	PROTECT(set_name=allocVector(VECSXP,2));
	PROTECT(set_name_r=allocVector(VECSXP,TreeWidth));

	//set column names
	SET_VECTOR_ELT(set_name_r, 0, mkChar("NodeType")); 	// (0: unused info matrix space; 1: internal node; 2: terminal node)
	SET_VECTOR_ELT(set_name_r, 1, mkChar("Node"));		// node index
	SET_VECTOR_ELT(set_name_r, 2, mkChar("NumObs"));	// number of observations
	SET_VECTOR_ELT(set_name_r, 3, mkChar("NodeMean"));	// within node y mean
	SET_VECTOR_ELT(set_name_r, 4, mkChar("NextLeft"));	// left daughter node
	SET_VECTOR_ELT(set_name_r, 5, mkChar("NextRight"));	// right daughter node
	SET_VECTOR_ELT(set_name_r, 6, mkChar("NumOfComb"));	// number of x combination
	SET_VECTOR_ELT(set_name_r, 7, mkChar("SplitValue"));// splitting value

	char str[10];

	for (i = 0; i < combsplit; i ++)
	{
		sprintf(str, "SplitVar%i", i+1);
		SET_VECTOR_ELT(set_name_r, 8+i, mkChar(str)); 			// x var
		sprintf(str, "Loading%i", i+1);
		SET_VECTOR_ELT(set_name_r, 8+i+combsplit, mkChar(str));	// x loading
	}

	SET_VECTOR_ELT(set_name, 0, set_name_r);

	int tempTreeLength;

	// converting each tree into R subject
	for (nt = 0; nt<ntrees; nt++)
	{
		tempTreeLength = 0;  // calculate tree length

		while (tree_matrix[nt][tempTreeLength] != NULL)
			tempTreeLength ++;

		PROTECT(FittedOneTree = allocMatrix(REALSXP, TreeWidth, tempTreeLength));

		for (i = 0; i < tempTreeLength; i++) // copy tree matrix
		{
			for (j = 0; j < TreeWidth; j++)
				REAL(FittedOneTree)[j + i*TreeWidth] = tree_matrix[nt][i][j];

			free(tree_matrix[nt][i]); // free copied tree
		}

		free(tree_matrix[nt]);
		setAttrib(FittedOneTree, R_DimNamesSymbol, set_name);
		SET_VECTOR_ELT(FittedTrees_R, nt, FittedOneTree);
		UNPROTECT(1);
	}
	free(tree_matrix);

	// variable importance
	SEXP AllError_R = R_NilValue;
	SEXP VarImp_R = R_NilValue;

	if (importance)
	{
		PROTECT(VarImp_R = allocMatrix(REALSXP, 1, dataX_p));
		PROTECT(AllError_R = allocMatrix(REALSXP, ntrees, dataX_p+1));

		for (j=0; j< dataX_p; j++) REAL(VarImp_R)[j] = VarImp[j];
		free(VarImp);

		for (nt=0; nt<ntrees; nt++)
		{
			for (j=0; j< dataX_p+1; j++)
			{
				REAL(AllError_R)[nt + j * ntrees] = AllError[nt][j];
			}
			free(AllError[nt]);
		}
		free(AllError);
	}

	// observation track
	SEXP ObsTrack_R = R_NilValue;
 	if (track_obs)
	{
		PROTECT(ObsTrack_R = allocMatrix(REALSXP, data_n, ntrees));
		for (nt=0; nt<ntrees; nt++)
		{
			for (i=0; i<data_n; i++)
				REAL(ObsTrack_R)[i + nt*data_n] = ObsTrack[nt][i];
			free(ObsTrack[nt]);
		}
		free(ObsTrack);
	}

	// create R object for return
	SEXP list_names;
	PROTECT(list_names = allocVector(STRSXP, 4));
	SET_STRING_ELT(list_names, 0, mkChar("FittedTrees"));
	SET_STRING_ELT(list_names, 1, mkChar("AllError"));
	SET_STRING_ELT(list_names, 2, mkChar("VarImp"));
	SET_STRING_ELT(list_names, 3, mkChar("ObsTrack"));

	SEXP FittedModel;

	PROTECT(FittedModel = allocVector(VECSXP, 4));

	SET_VECTOR_ELT(FittedModel, 0, FittedTrees_R);
	SET_VECTOR_ELT(FittedModel, 1, AllError_R);
	SET_VECTOR_ELT(FittedModel, 2, VarImp_R);
	SET_VECTOR_ELT(FittedModel, 3, ObsTrack_R);

	setAttrib(FittedModel, R_NamesSymbol, list_names);

	// unprotect R objects
	UNPROTECT(5 + 2*importance + track_obs);

	return FittedModel;
}

SEXP RLT_regression_predict(SEXP datasetX_R,
					SEXP FittedTrees_R,
					SEXP ncat_R,
					SEXP parameters_int_R,
					SEXP parameters_double_R)
{

	PARAMETERS *myPara = malloc(sizeof(PARAMETERS));
	copyParameters(myPara, parameters_int_R, parameters_double_R);

	SEXP dataX_dim = getAttrib(datasetX_R, R_DimSymbol);
	myPara->data_n = INTEGER(dataX_dim)[0];
	myPara->dataX_p = INTEGER(dataX_dim)[1];

	int data_n = myPara->data_n;
	int dataX_p = myPara->dataX_p;
	int combsplit = myPara->combsplit;
	int ntrees = myPara->ntrees;
	int i;
	int j;

	// creating tree indices

	int nt = 0;

	double ***tree_matrix = (double ***) malloc(ntrees * sizeof(double **));

	int TreeWidth = 8 + 2*combsplit;
    int OneTreeLength;

	for (nt = 0; nt < ntrees; nt++)
	{
		OneTreeLength = INTEGER(getAttrib(VECTOR_ELT(FittedTrees_R, nt), R_DimSymbol))[1];
		tree_matrix[nt] = (double **) malloc(OneTreeLength * sizeof(double *));

		for (i = 0; i < OneTreeLength; i++)
			tree_matrix[nt][i] = &REAL(VECTOR_ELT(FittedTrees_R, nt))[i*TreeWidth];
	}

	// get data index
	double **x = (double **) malloc(dataX_p * sizeof(double *));
	for (j = 0; j < dataX_p; j++)
		x[j] = &REAL(datasetX_R)[j*data_n];

	int* ncat = INTEGER(ncat_R);

	// get predictions

	double** AllPrediction = (double **) malloc(ntrees * sizeof(double*));
	double* Prediction = (double *) calloc(data_n, sizeof(double));

	int* obsInd = (int *) malloc(data_n * sizeof(int));
	for (i = 0; i < data_n; i++)
		obsInd[i] = i;

	for (nt=0; nt<ntrees; nt++)
	{
		AllPrediction[nt] = (double *) malloc(data_n * sizeof(double));
		predict_reg(0, obsInd, x, tree_matrix[nt], combsplit, ncat, obsInd, AllPrediction[nt], data_n);

		for (i=0; i<data_n; i++)
		{
			Prediction[i] += AllPrediction[nt][i];
		}
	}

	for (i=0; i<data_n; i++)
		Prediction[i] /= ntrees;

	SEXP AllPrediction_R;
	SEXP Prediction_R;

	PROTECT(AllPrediction_R = allocMatrix(REALSXP, data_n, ntrees));
	PROTECT(Prediction_R = allocVector(REALSXP, data_n));

	for (i = 0; i< data_n; i++)
	{
		REAL(Prediction_R)[i] = Prediction[i];

		for (nt = 0; nt < ntrees; nt++)
			REAL(AllPrediction_R)[nt*data_n + i] = AllPrediction[nt][i];
	}

	SEXP list_names;

	PROTECT(list_names = allocVector(STRSXP, 2));
	SET_STRING_ELT(list_names, 0, mkChar("AllPrediction"));
	SET_STRING_ELT(list_names, 1, mkChar("Prediction"));

	SEXP Ypredict;

	PROTECT(Ypredict = allocVector(VECSXP, 2));

	SET_VECTOR_ELT(Ypredict, 0, AllPrediction_R);
	SET_VECTOR_ELT(Ypredict, 1, Prediction_R);
	setAttrib(Ypredict, R_NamesSymbol, list_names);

	free(Prediction);

	for (nt=0; nt<ntrees; nt++)
		free(AllPrediction[nt]);
	free(AllPrediction);
	free(x);
	free(obsInd);
	for (nt=0; nt<ntrees; nt++)
		free(tree_matrix[nt]);
	free(tree_matrix);
	free(myPara);

	UNPROTECT(4);

	return(Ypredict);
}

// classification

SEXP RLT_classification(SEXP datasetX_R,
						SEXP datasetY_R,
						SEXP ncat_R,
						SEXP subjectweight_R,
						SEXP variableweight_R,
						SEXP parameters_int_R,
						SEXP parameters_double_R)
{
	// R_CStackLimit=(uintptr_t)-1;  // raise the stack limit of R

	// get data dimension and parameters

	PARAMETERS *myPara = malloc(sizeof(PARAMETERS));
	copyParameters(myPara, parameters_int_R, parameters_double_R);

	SEXP dataX_dim = getAttrib(datasetX_R, R_DimSymbol);
	myPara->model = 2;
	myPara->data_n = INTEGER(dataX_dim)[0];
	myPara->dataX_p = INTEGER(dataX_dim)[1];

	if (myPara->summary >= 1)
	{
		Rprintf("Reinforcement Learning Trees for regression, Version 2.0 \n");
		printParameters(myPara);
	}

	int data_n = myPara->data_n;
	int dataX_p = myPara->dataX_p;
	int nclass = myPara->nclass;
	int i;
	int j;
	int nt = 0;

	int ntrees = myPara->ntrees;
	int combsplit = myPara->combsplit;
	int importance = myPara->importance;
	int track_obs = myPara->track_obs;

	// get data
	double **dataX_matrix = (double **) malloc(dataX_p * sizeof(double *));
	for (j = 0; j < dataX_p; j++)
		dataX_matrix[j] = &REAL(datasetX_R)[j*data_n];

	int *dataY_vector = INTEGER(datasetY_R);
	int *ncat = INTEGER(ncat_R);

	// copy subjectweight
	double *subjectweight = REAL(subjectweight_R);
	standardize(subjectweight, data_n);	// this could change the input data due to precision loss...

	double *variableweight = REAL(variableweight_R);
	standardize(variableweight, dataX_p);	// this could change the input data due to precision loss...

	// copy partialdata
	int *partialdata = (int *) malloc (data_n * sizeof(int));
	if (partialdata == NULL) error("Unable to malloc for observation index");
	for (i = 0; i < data_n; i++)
		partialdata[i] = i;

	// variables can be used
	int *usevariable = (int *) malloc (dataX_p * sizeof(int));
	if (usevariable == NULL) error("Unable to malloc for variable index");
	for (j = 0; j < dataX_p; j++)
		usevariable[j] = j;

	// protected variables
	int *protectvariable = (int *) malloc (dataX_p * sizeof(int));
	if (protectvariable == NULL) error("Unable to malloc for variable index");
	for (j = 0; j < dataX_p; j++)
		protectvariable[j] = 0;

	// get tree matrix
	int TreeMaxLength = 1 + 2*data_n; // 3*imax((int) data_n/nmin, 1) + 1 + 20;  // if require minimum sample size
	int TreeWidth = 8 + 2*combsplit + nclass;

	// create matrices for fitted trees
	double ***tree_matrix = (double ***) malloc(ntrees * sizeof(double **));
	if (tree_matrix == NULL) error("Unable to malloc for tree matrix");

	for (nt=0; nt<ntrees; nt++)
	{
		tree_matrix[nt] = (double **) malloc(TreeMaxLength * sizeof(double*));
		if (tree_matrix[nt] == NULL) error("Unable to malloc for tree matrix");

		for (i = 0; i < TreeMaxLength; i++)
			tree_matrix[nt][i] = NULL; // if this is NULL, then is node is not used yet
	}

	// variable importance for each tree
	double **AllError = (double **) malloc(ntrees * sizeof(double *));
	if (AllError == NULL) error("Unable to calloc for mse recording");

	double *VarImp = calloc(dataX_p, sizeof(double));
	if (VarImp == NULL) error("Unable to calloc for variable importance");

	if (importance)
		for (nt=0; nt<ntrees; nt++)
		{
			AllError[nt] = (double *) calloc((dataX_p+1), sizeof(double));
			if (AllError[nt] == NULL) error("Unable to calloc for mse recording");
		}

	// observation registration
	int **ObsTrack = (int **) malloc(ntrees * sizeof(int *));
	if (track_obs)
		for (nt = 0; nt<ntrees; nt++)
		{
			ObsTrack[nt] = (int *) calloc(data_n, sizeof(int));
			if (ObsTrack[nt] == NULL) error("Unable to calloc for observation track");
		}
	////////////////////////////////////////////////////////////////////////////
	///////////     Start to fit RLT model here     ////////////////////////////
	////////////////////////////////////////////////////////////////////////////

	if (myPara->summary >= 2)
		Rprintf("--------- start to fit RLT trees ------------ \n");

/* 	FILE * Output;
	Output = fopen("error.txt", "w+");
	fprintf(Output, " --- Start fitting trees --- \n");
	fclose(Output); */

  GetRNGstate();
	Fit_Trees_classification(dataX_matrix, dataY_vector, tree_matrix, AllError, VarImp, ObsTrack, myPara, ncat, subjectweight, variableweight, partialdata, usevariable, protectvariable, data_n, dataX_p);
	PutRNGstate();

/* 	Output = fopen("error.txt", "a");
	fprintf(Output, " --- end fitting trees --- \n");
	fclose(Output);	 */

	// free some memory that is not used in the output
	free(dataX_matrix);
	free(partialdata);
	free(usevariable);
	free(protectvariable);
	free(myPara);

	// converting tree_matrix into R output

	SEXP FittedTrees_R;
	SEXP FittedOneTree;

	PROTECT(FittedTrees_R = allocVector(VECSXP, ntrees));

	SEXP set_name;  // both column and row names
	SEXP set_name_r; // just column names

	PROTECT(set_name=allocVector(VECSXP,2));
	PROTECT(set_name_r=allocVector(VECSXP,TreeWidth));

	//set column names
	SET_VECTOR_ELT(set_name_r, 0, mkChar("NodeType")); 	// (0: unused info matrix space; 1: internal node; 2: terminal node)
	SET_VECTOR_ELT(set_name_r, 1, mkChar("Node"));		// node index
	SET_VECTOR_ELT(set_name_r, 2, mkChar("NumObs"));	// number of observations
	SET_VECTOR_ELT(set_name_r, 3, mkChar("BestClass"));	// within node y mean
	SET_VECTOR_ELT(set_name_r, 4, mkChar("NextLeft"));	// left daughter node
	SET_VECTOR_ELT(set_name_r, 5, mkChar("NextRight"));	// right daughter node
	SET_VECTOR_ELT(set_name_r, 6, mkChar("NumOfComb"));	// number of x combination
	SET_VECTOR_ELT(set_name_r, 7, mkChar("SplitValue"));// splitting value

	char str[10];

	for (i = 0; i < combsplit; i ++)
	{
		sprintf(str, "SplitVar%i", i+1);
		SET_VECTOR_ELT(set_name_r, 8+i, mkChar(str)); 			// x var
		sprintf(str, "Loading%i", i+1);
		SET_VECTOR_ELT(set_name_r, 8+i+combsplit, mkChar(str));	// x loading
	}

	for (i = 0; i < nclass; i ++)
	{
		sprintf(str, "Class%i", i);
		SET_VECTOR_ELT(set_name_r, 8 + 2*combsplit + i, mkChar(str)); // y class
	}

	SET_VECTOR_ELT(set_name, 0, set_name_r);

	int tempTreeLength;

	// converting each tree into R subject
	for (nt = 0; nt<ntrees; nt++)
	{
		tempTreeLength = 0;  // calculate tree length

		while (tree_matrix[nt][tempTreeLength] != NULL)
			tempTreeLength ++;

		PROTECT(FittedOneTree = allocMatrix(REALSXP, TreeWidth, tempTreeLength));

		for (i = 0; i < tempTreeLength; i++) // copy tree matrix
		{
			for (j = 0; j < TreeWidth; j++)
				REAL(FittedOneTree)[j + i*TreeWidth] = tree_matrix[nt][i][j];

			free(tree_matrix[nt][i]); // free copied tree
		}

		free(tree_matrix[nt]);
		setAttrib(FittedOneTree, R_DimNamesSymbol, set_name);
		SET_VECTOR_ELT(FittedTrees_R, nt, FittedOneTree);
		UNPROTECT(1);
	}
	free(tree_matrix);

	// variable importance
	SEXP AllError_R = R_NilValue;
	SEXP VarImp_R = R_NilValue;

	if (importance)
	{
		PROTECT(VarImp_R = allocMatrix(REALSXP, 1, dataX_p));
		PROTECT(AllError_R = allocMatrix(REALSXP, ntrees, dataX_p+1));

		for (j=0; j< dataX_p; j++) REAL(VarImp_R)[j] = VarImp[j];
		free(VarImp);

		for (nt=0; nt<ntrees; nt++)
		{
			for (j=0; j< dataX_p+1; j++)
			{
				REAL(AllError_R)[nt + j * ntrees] = AllError[nt][j];
			}
			free(AllError[nt]);
		}
		free(AllError);
	}

	// observation track
	SEXP ObsTrack_R = R_NilValue;
 	if (track_obs)
	{
		PROTECT(ObsTrack_R = allocMatrix(REALSXP, data_n, ntrees));
		for (nt=0; nt<ntrees; nt++)
		{
			for (i=0; i<data_n; i++)
				REAL(ObsTrack_R)[i + nt*data_n] = ObsTrack[nt][i];
			free(ObsTrack[nt]);
		}
		free(ObsTrack);
	}

	// create R object for return
	SEXP list_names;
	PROTECT(list_names = allocVector(STRSXP, 4));
	SET_STRING_ELT(list_names, 0, mkChar("FittedTrees"));
	SET_STRING_ELT(list_names, 1, mkChar("AllError"));
	SET_STRING_ELT(list_names, 2, mkChar("VarImp"));
	SET_STRING_ELT(list_names, 3, mkChar("ObsTrack"));

	SEXP FittedModel;

	PROTECT(FittedModel = allocVector(VECSXP, 4));

	SET_VECTOR_ELT(FittedModel, 0, FittedTrees_R);
	SET_VECTOR_ELT(FittedModel, 1, AllError_R);
	SET_VECTOR_ELT(FittedModel, 2, VarImp_R);
	SET_VECTOR_ELT(FittedModel, 3, ObsTrack_R);

	setAttrib(FittedModel, R_NamesSymbol, list_names);

	// unprotect R objects
	UNPROTECT(5 + 2*importance + track_obs);

	return FittedModel;
}

SEXP RLT_classification_predict(SEXP datasetX_R,
								SEXP FittedTrees_R,
								SEXP ncat_R,
								SEXP parameters_int_R,
								SEXP parameters_double_R)
{

	PARAMETERS *myPara = malloc(sizeof(PARAMETERS));
	copyParameters(myPara, parameters_int_R, parameters_double_R);

	SEXP dataX_dim = getAttrib(datasetX_R, R_DimSymbol);
	myPara->data_n = INTEGER(dataX_dim)[0];
	myPara->dataX_p = INTEGER(dataX_dim)[1];

	int data_n = myPara->data_n;
	int dataX_p = myPara->dataX_p;
	int combsplit = myPara->combsplit;
	int ntrees = myPara->ntrees;
	int nclass = myPara->nclass;
	int i;
	int j;

	// creating tree indices

	int nt = 0;

	double ***tree_matrix = (double ***) malloc(ntrees * sizeof(double **));

	int TreeWidth = 8 + 2*combsplit + nclass;
    int OneTreeLength;

	for (nt = 0; nt < ntrees; nt++)
	{
		OneTreeLength = INTEGER(getAttrib(VECTOR_ELT(FittedTrees_R, nt), R_DimSymbol))[1];
		tree_matrix[nt] = (double **) malloc(OneTreeLength * sizeof(double *));

		for (i = 0; i < OneTreeLength; i++)
			tree_matrix[nt][i] = &REAL(VECTOR_ELT(FittedTrees_R, nt))[i*TreeWidth];
	}

	// get data index
	double **x = (double **) malloc(dataX_p * sizeof(double *));
	for (j = 0; j < dataX_p; j++)
		x[j] = &REAL(datasetX_R)[j*data_n];

	int* ncat = INTEGER(ncat_R);

	// get predictions

	double*** AllPrediction = (double ***) malloc(ntrees * sizeof(double**));
	int* Prediction = (int *) calloc(data_n, sizeof(int));

	int* obsInd = (int *) malloc(data_n * sizeof(int));
	for (i = 0; i < data_n; i++)
		obsInd[i] = i;

	for (nt=0; nt<ntrees; nt++)
	{
		AllPrediction[nt] = (double **) malloc(data_n * sizeof(double*));

		for (i = 0; i < data_n; i++)
			AllPrediction[nt][i] = (double*) malloc(nclass * sizeof(double));

		predict_cla_all(0, obsInd, x, tree_matrix[nt], nclass, combsplit, ncat, obsInd, AllPrediction[nt], data_n);
	}

	double** ProbPrediction = (double **) malloc(data_n * sizeof(double*));
	for (i = 0; i < data_n; i++)
		ProbPrediction[i] = calloc(nclass, sizeof(double));

	double tempprob;

	for (i=0; i<data_n; i++)
	{
		for (nt=0; nt<ntrees; nt++)
			for (j=0; j<nclass; j++)
				ProbPrediction[i][j] += AllPrediction[nt][i][j];

		for (j=0; j<nclass; j++)
			ProbPrediction[i][j] /= nt;

		tempprob = ProbPrediction[i][0];
		Prediction[i] = 0;

		for (j=0; j<nclass; j++)
		{
			if (ProbPrediction[i][j] > tempprob)
			{
				tempprob = ProbPrediction[i][j];
				Prediction[i] = j;
			}
		}
	}

	SEXP AllPrediction_R;
	SEXP ProbPrediction_R;
	SEXP Prediction_R;

	PROTECT(AllPrediction_R = allocVector(VECSXP, ntrees));
	PROTECT(ProbPrediction_R = allocMatrix(REALSXP, data_n, nclass));
	PROTECT(Prediction_R = allocVector(INTSXP, data_n));

	SEXP OneTreePred;

	for (nt = 0; nt< ntrees; nt++)
	{
		PROTECT(OneTreePred = allocMatrix(REALSXP, data_n, nclass));

		for (i = 0; i< data_n; i++)
		{
			for (j = 0; j < nclass; j++)
				REAL(OneTreePred)[j*data_n + i] = AllPrediction[nt][i][j];
			free(AllPrediction[nt][i]);
		}

		free(AllPrediction[nt]);

		SET_VECTOR_ELT(AllPrediction_R, nt, OneTreePred);
		UNPROTECT(1);
	}
	free(AllPrediction);

	for (i = 0; i< data_n; i++)
	{
		for (j = 0; j < nclass; j++)
			REAL(ProbPrediction_R)[j*data_n + i] = ProbPrediction[i][j];

		free(ProbPrediction[i]);

		INTEGER(Prediction_R)[i] = Prediction[i];
	}
	free(ProbPrediction);
	free(Prediction);

	SEXP list_names;

	PROTECT(list_names = allocVector(STRSXP, 3));
	SET_STRING_ELT(list_names, 0, mkChar("AllPrediction"));
	SET_STRING_ELT(list_names, 1, mkChar("ProbPrediction"));
	SET_STRING_ELT(list_names, 2, mkChar("Prediction"));

	SEXP Ypredict;

	PROTECT(Ypredict = allocVector(VECSXP, 3));

	SET_VECTOR_ELT(Ypredict, 0, AllPrediction_R);
	SET_VECTOR_ELT(Ypredict, 1, ProbPrediction_R);
	SET_VECTOR_ELT(Ypredict, 2, Prediction_R);
	setAttrib(Ypredict, R_NamesSymbol, list_names);

	free(x);
	free(obsInd);
	for (nt=0; nt<ntrees; nt++)
		free(tree_matrix[nt]);
	free(tree_matrix);
	free(myPara);
	UNPROTECT(5);

	return(Ypredict);
}

// survival

SEXP RLT_survival(SEXP datasetX_R,
					SEXP datasetY_R,
					SEXP datasetCensor_R,
					SEXP datasetInterval_R,
					SEXP ncat_R,
					SEXP subjectweight_R,
					SEXP variableweight_R,
					SEXP parameters_int_R,
					SEXP parameters_double_R)
{
	// R_CStackLimit=(uintptr_t)-1;  // raise the stack limit of R

	// get data dimension and parameters

	PARAMETERS *myPara = malloc(sizeof(PARAMETERS));
	copyParameters(myPara, parameters_int_R, parameters_double_R);

	SEXP dataX_dim = getAttrib(datasetX_R, R_DimSymbol);
	myPara->model = 3;
	myPara->data_n = INTEGER(dataX_dim)[0];
	myPara->dataX_p = INTEGER(dataX_dim)[1];

	if (myPara->summary >= 1)
	{
		Rprintf("Reinforcement Learning Trees for regression, Version 2.0 \n");
		printParameters(myPara);
	}

	int data_n = myPara->data_n;
	int dataX_p = myPara->dataX_p;
	int i;
	int j;
	int nt = 0;

	int ntrees = myPara->ntrees;
	int combsplit = myPara->combsplit;
	int importance = myPara->importance;
	int track_obs = myPara->track_obs;

	// get data
	double **dataX_matrix = (double **) malloc(dataX_p * sizeof(double *));
	for (j = 0; j < dataX_p; j++)
		dataX_matrix[j] = &REAL(datasetX_R)[j*data_n];

	int *dataY_vector = INTEGER(datasetY_R);
	double *dataInterval_vector = REAL(datasetInterval_R);

	// get number of unique failure times

	int nfail_unique = 0;
	for (i = 0; i < data_n; i++)
		if (dataY_vector[i] > nfail_unique)
			nfail_unique = dataY_vector[i];

	int *dataCensor_vector = INTEGER(datasetCensor_R);

	int *ncat = INTEGER(ncat_R);

	// copy subjectweight
	double *subjectweight = REAL(subjectweight_R);
	standardize(subjectweight, data_n);	// this could change the input data due to precision loss...

	double *variableweight = REAL(variableweight_R);
	standardize(variableweight, dataX_p);	// this could change the input data due to precision loss...

	// copy partialdata
	int *partialdata = (int *) malloc (data_n * sizeof(int));
	if (partialdata == NULL) error("Unable to malloc for observation index");
	for (i = 0; i < data_n; i++)
		partialdata[i] = i;

	// variables can be used
	int *usevariable = (int *) malloc (dataX_p * sizeof(int));
	if (usevariable == NULL) error("Unable to malloc for variable index");
	for (j = 0; j < dataX_p; j++)
		usevariable[j] = j;

	// protected variables
	int *protectvariable = (int *) malloc (dataX_p * sizeof(int));
	if (protectvariable == NULL) error("Unable to malloc for variable index");
	for (j = 0; j < dataX_p; j++)
		protectvariable[j] = 0;

	// get tree matrix
	int TreeMaxLength = 1 + 2*data_n; // 3*imax((int) data_n/nmin, 1) + 1 + 20;  // if require minimum sample size
	int TreeWidth = 8 + 2*combsplit;

	// create matrices for fitted trees
	double ***tree_matrix = (double ***) malloc(ntrees * sizeof(double **));
	if (tree_matrix == NULL) error("Unable to malloc for tree matrix");

	for (nt=0; nt<ntrees; nt++)
	{
		tree_matrix[nt] = (double **) malloc(TreeMaxLength * sizeof(double*));
		if (tree_matrix[nt] == NULL) error("Unable to malloc for tree matrix");

		for (i = 0; i < TreeMaxLength; i++)
			tree_matrix[nt][i] = NULL; // if this is NULL, then is node is not used yet
	}

	// create matrices for fitted survival curve

	double ***surv_matrix = (double ***) malloc(ntrees * sizeof(double **));
	if (surv_matrix == NULL) error("Unable to malloc for tree matrix");

	for (nt=0; nt<ntrees; nt++)
	{
		surv_matrix[nt] = (double **) malloc(nfail_unique * sizeof(double*)); // At most nfail_unique terminal nodes
		if (surv_matrix[nt] == NULL) error("Unable to malloc for tree matrix");

		for (i = 0; i < nfail_unique; i++)
			surv_matrix[nt][i] = NULL; // if this is NULL, then is node is not used yet
	}

	// variable importance for each tree
	double **AllError = (double **) malloc(ntrees * sizeof(double *));
	if (AllError == NULL) error("Unable to calloc for mse recording");

	double *VarImp = calloc(dataX_p, sizeof(double));
	if (VarImp == NULL) error("Unable to calloc for variable importance");

	if (importance)
		for (nt=0; nt<ntrees; nt++)
		{
			AllError[nt] = (double *) calloc((dataX_p+1), sizeof(double));
			if (AllError[nt] == NULL) error("Unable to calloc for mse recording");
		}

	// observation registration
	int **ObsTrack = (int **) malloc(ntrees * sizeof(int *));
	if (track_obs)
		for (nt = 0; nt<ntrees; nt++)
		{
			ObsTrack[nt] = (int *) calloc(data_n, sizeof(int));
			if (ObsTrack[nt] == NULL) error("Unable to calloc for observation track");
		}
	////////////////////////////////////////////////////////////////////////////
	///////////     Start to fit RLT model here     ////////////////////////////
	////////////////////////////////////////////////////////////////////////////

	if (myPara->summary >= 2)
		Rprintf("--------- start to fit RLT trees ------------ \n");

/*  FILE * Output;
	Output = fopen("error.txt", "w+");
	fprintf(Output, " --- Start fitting trees --- \n");
	fclose(Output);  */

  GetRNGstate();
	Fit_Trees_survival(dataX_matrix, dataY_vector, dataCensor_vector, dataInterval_vector, tree_matrix, surv_matrix, AllError, VarImp, ObsTrack, myPara, ncat, subjectweight, variableweight, partialdata, usevariable, protectvariable, data_n, dataX_p, nfail_unique);
	PutRNGstate();

/* 	Output = fopen("error.txt", "a");
	fprintf(Output, " --- end fitting trees --- \n");
	fclose(Output);	 */

	// free some memory that is not used in the output
	free(dataX_matrix);
	free(partialdata);
	free(usevariable);
	free(protectvariable);
	free(myPara);

	// converting tree_matrix into R output

	SEXP FittedTrees_R;
	SEXP FittedSurv_R;
	SEXP FittedOneTree;

	PROTECT(FittedTrees_R = allocVector(VECSXP, ntrees));
    PROTECT(FittedSurv_R = allocVector(VECSXP, ntrees));

	SEXP set_name;  // both column and row names
	SEXP set_name_r; // just column names

	PROTECT(set_name=allocVector(VECSXP,2));
	PROTECT(set_name_r=allocVector(VECSXP,TreeWidth));

	//set column names
	SET_VECTOR_ELT(set_name_r, 0, mkChar("NodeType")); 	// (0: unused info matrix space; 1: internal node; 2: terminal node)
	SET_VECTOR_ELT(set_name_r, 1, mkChar("Node"));		// node index
	SET_VECTOR_ELT(set_name_r, 2, mkChar("NumObs"));	// number of observations
	SET_VECTOR_ELT(set_name_r, 3, mkChar("SurvivalNode"));	// corresponding node in surv_matrix
	SET_VECTOR_ELT(set_name_r, 4, mkChar("NextLeft"));	// left daughter node
	SET_VECTOR_ELT(set_name_r, 5, mkChar("NextRight"));	// right daughter node
	SET_VECTOR_ELT(set_name_r, 6, mkChar("NumOfComb"));	// number of x combination
	SET_VECTOR_ELT(set_name_r, 7, mkChar("SplitValue"));// splitting value

	char str[10];

	for (i = 0; i < combsplit; i ++)
	{
		sprintf(str, "SplitVar%i", i+1);
		SET_VECTOR_ELT(set_name_r, 8+i, mkChar(str)); 			// x var
		sprintf(str, "Loading%i", i+1);
		SET_VECTOR_ELT(set_name_r, 8+i+combsplit, mkChar(str));	// x loading
	}

	SET_VECTOR_ELT(set_name, 0, set_name_r);

	int tempTreeLength;

	// converting each tree into R subject

	for (nt = 0; nt<ntrees; nt++)
	{
		tempTreeLength = 0;  // calculate tree length

		while (tree_matrix[nt][tempTreeLength] != NULL)
			tempTreeLength ++;

		PROTECT(FittedOneTree = allocMatrix(REALSXP, TreeWidth, tempTreeLength));


		for (i = 0; i < tempTreeLength; i++) // copy tree matrix
		{
			for (j = 0; j < TreeWidth; j++)
				REAL(FittedOneTree)[j + i*TreeWidth] = tree_matrix[nt][i][j];

			free(tree_matrix[nt][i]); // free copied tree
		}

		free(tree_matrix[nt]);
		setAttrib(FittedOneTree, R_DimNamesSymbol, set_name);
		SET_VECTOR_ELT(FittedTrees_R, nt, FittedOneTree);
		UNPROTECT(1);
	}
	free(tree_matrix);

	// converting the terminal node survival functions into R subject

	for (nt = 0; nt<ntrees; nt++)
	{
		tempTreeLength = 0;  // calculate tree length

		while (surv_matrix[nt][tempTreeLength] != NULL)
			tempTreeLength ++;

		PROTECT(FittedOneTree = allocMatrix(REALSXP, nfail_unique+1, tempTreeLength));

		// copy tree matrix
		for (i = 0; i < tempTreeLength; i++)
		{
			for (j = 0; j < (nfail_unique+1); j++)
				REAL(FittedOneTree)[j + i*(nfail_unique+1)] = surv_matrix[nt][i][j];

			free(surv_matrix[nt][i]);
		}

		free(surv_matrix[nt]);
		SET_VECTOR_ELT(FittedSurv_R, nt, FittedOneTree);
		UNPROTECT(1);
	}
	free(surv_matrix);

	// variable importance
	SEXP AllError_R = R_NilValue;
	SEXP VarImp_R = R_NilValue;

	if (importance)
	{
		PROTECT(VarImp_R = allocMatrix(REALSXP, 1, dataX_p));
		PROTECT(AllError_R = allocMatrix(REALSXP, ntrees, dataX_p+1));

		for (j=0; j< dataX_p; j++) REAL(VarImp_R)[j] = VarImp[j];
		free(VarImp);

		for (nt=0; nt<ntrees; nt++)
		{
			for (j=0; j< dataX_p+1; j++)
			{
				REAL(AllError_R)[nt + j * ntrees] = AllError[nt][j];
			}
			free(AllError[nt]);
		}
		free(AllError);
	}

	// observation track
	SEXP ObsTrack_R = R_NilValue;
 	if (track_obs)
	{
		PROTECT(ObsTrack_R = allocMatrix(REALSXP, data_n, ntrees));
		for (nt=0; nt<ntrees; nt++)
		{
			for (i=0; i<data_n; i++)
				REAL(ObsTrack_R)[i + nt*data_n] = ObsTrack[nt][i];
			free(ObsTrack[nt]);
		}
		free(ObsTrack);
	}

	// create R object for return
	SEXP list_names;
	PROTECT(list_names = allocVector(STRSXP, 5));
	SET_STRING_ELT(list_names, 0, mkChar("FittedTrees"));
	SET_STRING_ELT(list_names, 1, mkChar("FittedSurv"));
	SET_STRING_ELT(list_names, 2, mkChar("AllError"));
	SET_STRING_ELT(list_names, 3, mkChar("VarImp"));
	SET_STRING_ELT(list_names, 4, mkChar("ObsTrack"));

	SEXP FittedModel;

	PROTECT(FittedModel = allocVector(VECSXP, 5));

	SET_VECTOR_ELT(FittedModel, 0, FittedTrees_R);
	SET_VECTOR_ELT(FittedModel, 1, FittedSurv_R);
	SET_VECTOR_ELT(FittedModel, 2, AllError_R);
	SET_VECTOR_ELT(FittedModel, 3, VarImp_R);
	SET_VECTOR_ELT(FittedModel, 4, ObsTrack_R);

	setAttrib(FittedModel, R_NamesSymbol, list_names);

	// unprotect R objects
	UNPROTECT(6 + 2*importance + track_obs);

	return FittedModel;
}



SEXP RLT_survival_predict(SEXP datasetX_R,
							SEXP FittedTrees_R,
							SEXP FittedSurv_R,
							SEXP ncat_R,
							SEXP parameters_int_R,
							SEXP parameters_double_R)
{
	PARAMETERS *myPara = malloc(sizeof(PARAMETERS));
	copyParameters(myPara, parameters_int_R, parameters_double_R);

	SEXP dataX_dim = getAttrib(datasetX_R, R_DimSymbol);
	myPara->data_n = INTEGER(dataX_dim)[0];
	myPara->dataX_p = INTEGER(dataX_dim)[1];

	int data_n = myPara->data_n;
	int dataX_p = myPara->dataX_p;
	int combsplit = myPara->combsplit;
	int ntrees = myPara->ntrees;
	int i;
	int j;

	// creating tree indices

	int nt = 0;

	double ***tree_matrix = (double ***) malloc(ntrees * sizeof(double **));

	int TreeWidth = 8 + 2*combsplit;
    int OneTreeLength;

	for (nt = 0; nt < ntrees; nt++)
	{
		OneTreeLength = INTEGER(getAttrib(VECTOR_ELT(FittedTrees_R, nt), R_DimSymbol))[1];
		tree_matrix[nt] = (double **) malloc(OneTreeLength * sizeof(double *));

		for (i = 0; i < OneTreeLength; i++)
			tree_matrix[nt][i] = &REAL(VECTOR_ELT(FittedTrees_R, nt))[i*TreeWidth];
	}

	// get data index
	double **x = (double **) malloc(dataX_p * sizeof(double *));
	for (j = 0; j < dataX_p; j++)
		x[j] = &REAL(datasetX_R)[j*data_n];

	int* ncat = INTEGER(ncat_R);
	int nfail = INTEGER(getAttrib(VECTOR_ELT(FittedSurv_R, 0), R_DimSymbol))[0] - 1;
	// Rprintf("number of points is %i \n", nfail);

	double ***surv_matrix = (double ***) malloc(ntrees * sizeof(double **));

	for (nt = 0; nt < ntrees; nt++)
	{
		OneTreeLength = INTEGER(getAttrib(VECTOR_ELT(FittedSurv_R, nt), R_DimSymbol))[1];
		surv_matrix[nt] = (double **) malloc(OneTreeLength * sizeof(double *));

		for (i = 0; i < OneTreeLength; i++)
			surv_matrix[nt][i] = &REAL(VECTOR_ELT(FittedSurv_R, nt))[i*(nfail+1)];
	}

	// get predictions

	double** TempPred = (double **) malloc(data_n * sizeof(double*));
	double** SurvPred = (double **) malloc(data_n * sizeof(double*));
	for (i = 0; i < data_n; i++) SurvPred[i] = (double *) calloc(nfail, sizeof(double));

	int* obsInd = (int *) malloc(data_n * sizeof(int));
	for (i = 0; i < data_n; i++)
		obsInd[i] = i;

	for (nt=0; nt<ntrees; nt++)
	{
		predict_surv(0, obsInd, x, tree_matrix[nt], surv_matrix[nt], combsplit, ncat, obsInd, TempPred, data_n);

 		for (i =0; i < data_n; i++)
			for (j = 0; j < nfail; j++)
				SurvPred[i][j] += TempPred[i][j+1];
	}

	for (i =0; i < data_n; i++)
		for (j = 0; j < nfail; j++)
			SurvPred[i][j] /= ntrees;

/* 	double Surv;

	for (i =0; i < data_n; i++)
	{
		Surv = 1;
		for (j = 0; j < nfail; j++)
		{
			Surv *= (1 - SurvPred[i][j]);
			SurvPred[i][j] = Surv;
		}
	} */

	SEXP SurvPred_R;
	PROTECT(SurvPred_R = allocMatrix(REALSXP, data_n, nfail));

	for (i = 0; i< data_n; i++)
		for (j = 0; j < nfail; j++)
			REAL(SurvPred_R)[i + j*data_n] = SurvPred[i][j];

	for (i = 0; i< data_n; i++)
	{
		free(SurvPred[i]);
	}
	free(TempPred);


	SEXP list_names;

	PROTECT(list_names = allocVector(STRSXP, 1));
	SET_STRING_ELT(list_names, 0, mkChar("SurvPred"));

	SEXP Ypredict;

	PROTECT(Ypredict = allocVector(VECSXP, 1));

	SET_VECTOR_ELT(Ypredict, 0, SurvPred_R);

	setAttrib(Ypredict, R_NamesSymbol, list_names);

	free(x);
	free(obsInd);

	for (nt=0; nt<ntrees; nt++)
		free(tree_matrix[nt]);
	free(tree_matrix);

	for (nt=0; nt<ntrees; nt++)
		free(surv_matrix[nt]);
	free(surv_matrix);
	free(myPara);

	UNPROTECT(3);

	return(Ypredict);
}



