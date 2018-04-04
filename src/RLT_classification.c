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

void Fit_Trees_classification(double** dataX_matrix,
							  int* dataY_vector,
							  double*** tree_matrix,
							  double** AllError,
							  double* VarImp,
							  int** obs_in_trees,
							  PARAMETERS* myPara,
							  int* ncat,
							  double* subjectweight,
							  double* variableweight,
							  int* obsIndicator,
							  int* usevariable,
							  int* protectvariable,
							  int use_n,
							  int use_p)
{
	// get all needed parameters:

	int ntrees = myPara->ntrees;
	int nclass = myPara->nclass;
	int useCores = myPara->useCores;
	int summary = myPara->summary;
	int replacement = myPara->replacement;
	int track_obs = myPara->track_obs;
	int npermute = myPara->npermute;
	int combsplit = myPara->combsplit;
	int importance = myPara->importance;
	int use_sub_weight = myPara->use_sub_weight;
	double resample_prob = myPara->resample_prob;

	int resample_size = (int) use_n*resample_prob;
	int nt;
	int i, j, k;

	// parallel computing... set cores

	useCores = imin(ntrees, imax(1, useCores));

	if (useCores > 0) OMPMSG(1);

	int haveCores = omp_get_max_threads();

	if(useCores > haveCores)
	{
	  if (summary) Rprintf("Do not have %i cores, use maximum %i cores. \n", useCores, haveCores);
	  useCores = haveCores;
	}

	#pragma omp parallel private(nt, i, j, k) num_threads(useCores)
	{
		#pragma omp for schedule(guided)   // defines the chunk size
		for (nt = 0; nt < ntrees; nt++) // fit all trees
		{
/* 			FILE * Output;
			Output = fopen("error.txt", "a");
			fprintf(Output, "Start fitting tree %i with thread %i \n", nt, omp_get_thread_num());
			fclose(Output); */


			// in-bag and out-of-bag data indicator
			int *inbagObs = (int *) malloc(resample_size * sizeof(int));
			int *oobagObs = (int *) malloc(use_n * sizeof(int)); // initiate a longer one

			int OneSub;
			int oobag_n;

			for (i=0; i < use_n; i++)
				oobagObs[i] = obsIndicator[i];

			// sample in-bag observations
			if (replacement)
			{
				for (i = 0; i < resample_size; i++)
				{
					OneSub = random_in_range(0, use_n);
					inbagObs[i] = obsIndicator[OneSub];
					oobagObs[OneSub] = -1;
				}

				oobag_n = use_n;

				for (i=0; i<oobag_n; i++)
				{
					if (oobagObs[i] < 0)
					{
						oobagObs[i] = oobagObs[oobag_n-1];
						oobag_n --;
						i--;
					}
				}
			}else{
				for (i = 0; i < resample_size; i++)
				{
					OneSub = random_in_range(0, use_n-i);
					inbagObs[i] = oobagObs[OneSub];
					oobagObs[OneSub] = oobagObs[use_n-1-i];
				}
				oobag_n = use_n - resample_size;
			}

/* 			Rprintf("\n all observations are: \n");
			for (i=0; i< use_n; i++) Rprintf("  %i", obsIndicator[i]);

			Rprintf("\n inbag observations are: \n");
			for (i=0; i< resample_size; i++) Rprintf("  %i", inbagObs[i]);

			Rprintf("\n oobag observations are: \n");
			for (i=0; i< oobag_n; i++) Rprintf("  %i", oobagObs[i]); */

			tree_matrix[nt][0] = (double *) malloc( (8 + 2*combsplit + nclass) * sizeof(double));
			if (tree_matrix[nt][0] == NULL) error("Unable to malloc for tree matrix");

			Split_A_Node_classification(0, inbagObs, dataX_matrix, dataY_vector, tree_matrix[nt], myPara, ncat, subjectweight, variableweight, usevariable, protectvariable, resample_size, use_p);

			// summarize what observations are used in this tree
			if (track_obs)
			{
				for (i=0; i<resample_size; i++)
					obs_in_trees[nt][inbagObs[i]]++;
			}

			free(inbagObs);

			// summarize variable importance

			if (importance && oobag_n >= 2)
			{
				double totalweights = 0;

				if (use_sub_weight)
					for (i = 0; i<oobag_n; i++)
						totalweights += subjectweight[oobagObs[i]];

				int* Ypred = (int *) malloc (oobag_n * sizeof(int));
				int* Yind = (int *) malloc (oobag_n * sizeof(int));

				for (i=0; i<oobag_n; i++)
					Yind[i] = i;

				// non-premuted error
				predict_cla(0, Yind, dataX_matrix, tree_matrix[nt], combsplit, ncat, oobagObs, Ypred, oobag_n);

				if (use_sub_weight)
				{
					for (i = 0; i < oobag_n; i++)
						AllError[nt][use_p] += (Ypred[i] == dataY_vector[oobagObs[i]] ? 0 : subjectweight[oobagObs[i]]);

					AllError[nt][use_p] /= totalweights;
				}else{
					for (i = 0; i < oobag_n; i++)
						AllError[nt][use_p] += (Ypred[i] == dataY_vector[oobagObs[i]] ? 0 : 1.0);

					AllError[nt][use_p] /= oobag_n;
				}

				int* permuteInt = (int *) malloc (oobag_n * sizeof(int));
				for (i=0; i<oobag_n; i++)
					permuteInt[i] = i;

				// permuted error
				for (k=1; k<= npermute; k++)
				{
					permute(permuteInt, oobag_n);

					for (j=0; j<use_p; j++)
					{
						predict_cla_pj(0, Yind, dataX_matrix, tree_matrix[nt], combsplit, ncat, oobagObs, Ypred, oobag_n, permuteInt, usevariable[j]);

						if (use_sub_weight)
						{
							for (i = 0; i < oobag_n; i++)
								AllError[nt][j] += (Ypred[i] == dataY_vector[oobagObs[i]] ? 0 : subjectweight[oobagObs[i]]);

						}else{
							for (i = 0; i < oobag_n; i++)
								AllError[nt][j] += (Ypred[i] == dataY_vector[oobagObs[i]] ? 0 : 1.0);

						}
					}
				}

				free(permuteInt);
				free(Ypred);
				free(Yind);

				for (j=0; j<use_p; j++)
				{
					if (use_sub_weight)
					{
						AllError[nt][j] /= totalweights*npermute;
					}else{
						AllError[nt][j] /= oobag_n*npermute;
					}
				}
			}

			free(oobagObs);

/* 			if (summary >= 2)
				if ((nt+1) % imax(1, (int)(ntrees/10)) == 0)
					Rprintf("%2.1f%% trees are done ...\n", (double) 100*(nt+1)/ntrees); */
		}

		// summarize variable importance

		#pragma omp barrier

		if (importance)
		{
			#pragma omp for
			for (j=0; j<use_p; j++) // calculate variable importance
			{
				k = 0;
				for (nt = 0; nt < ntrees; nt++)
				{
					if (AllError[nt][use_p] > 0)
					{
						VarImp[j] += AllError[nt][j]/AllError[nt][use_p];
						k++;
					}
				}

				if (k > 0)
				{
					VarImp[j] = VarImp[j]/ntrees - 1;
				}else{
					VarImp[j] = NAN;
				}
			}
		}
	}
}

void Split_A_Node_classification(int Node,
								 int* useObs,
								 double** dataX_matrix,
								 int* dataY_vector,
								 double** tree_matrix_nt,
								 PARAMETERS* myPara,
								 int* ncat,
								 double* subjectweight,
								 double* variableweight,
								 int* usevariable,
								 int* protectvariable,
								 int node_n,
								 int node_p)
{
	// Rprintf("At node %i ----------------------- \n", Node);

	int nmin = myPara->nmin;
	int combsplit = myPara->combsplit;
	int nclass = myPara->nclass;
	int use_sub_weight = myPara->use_sub_weight;
	int i;
	int PartWidth = 8 + 2*combsplit;
	double tempprob;
	int root = ((Node == 0) ? 1 : 0);

	// calculate node information

	tree_matrix_nt[Node][1] = Node;				// Node index
	tree_matrix_nt[Node][2] = node_n;			// Node size

	for (i=0; i<nclass; i++)
		tree_matrix_nt[Node][PartWidth +i] = 0;	// initialize

	if (use_sub_weight)							// Node sample mean
	{
		double totalweights = 0;
		for (i=0; i<node_n; i++)
		{
			tree_matrix_nt[Node][PartWidth + dataY_vector[useObs[i]]] += subjectweight[useObs[i]];
			totalweights += subjectweight[useObs[i]];
		}
		for (i=0; i<nclass; i++) tree_matrix_nt[Node][PartWidth + i] /= totalweights;
	}else{
		for (i=0; i<node_n; i++) tree_matrix_nt[Node][PartWidth + dataY_vector[useObs[i]]]++;
		for (i=0; i<nclass; i++) tree_matrix_nt[Node][PartWidth + i] /= node_n;
	}

	tempprob = tree_matrix_nt[Node][PartWidth];
	tree_matrix_nt[Node][3] = 0;

	for (i=1; i<nclass; i++)
	{
		if (tree_matrix_nt[Node][PartWidth +i] > tempprob)
		{
			tree_matrix_nt[Node][3] = i;
			tempprob = tree_matrix_nt[Node][PartWidth +i];
		}
	}

	if (node_n < nmin)							// Terminate this node
	{
TerminateThisNode:

		tree_matrix_nt[Node][0] = 2;			// Node type: Terminal

		tree_matrix_nt[Node][4] = NAN;			// Next Left
		tree_matrix_nt[Node][5] = NAN;			// Next Right
		tree_matrix_nt[Node][6] = NAN;			// Number of combination
		tree_matrix_nt[Node][7] = NAN;			// Splitting value

		for (i = 0; i < combsplit; i ++)		// Splitting variable(s) and loading(s)
		{
			tree_matrix_nt[Node][8+i] = NAN;
			tree_matrix_nt[Node][8+combsplit+i] = NAN;
		}
	}else										// Split this node
	{

		SplitRule* OneSplit = Find_A_Split_classification(useObs, dataX_matrix, dataY_vector, myPara, ncat, subjectweight, variableweight, usevariable, protectvariable, node_n, node_p, root);

		if (OneSplit->NCombinations == 0) // did not find linear combination, go back to terminate the node
		{
			free(OneSplit);
			goto TerminateThisNode;
		}

		tree_matrix_nt[Node][0] = 1;			// Node type: Internal

		int nextleft = Node;
		while (tree_matrix_nt[nextleft] != NULL)
			nextleft++;

		int nextright = nextleft + 1;
		while (tree_matrix_nt[nextright] != NULL)
			nextright++;

		tree_matrix_nt[Node][4] = nextleft;					// Next Left
		tree_matrix_nt[Node][5] = nextright;				// Next Right

		int i;
		int j;
		double SplitValue = OneSplit->SplitValue;
		int LeftSize = 0;
		int RightSize = 0;
		int* useObsLeft = (int *) malloc(node_n * sizeof(int));
		int* useObsRight = (int *) malloc(node_n * sizeof(int));

		tree_matrix_nt[Node][6] = OneSplit->NCombinations;	// Number of combination
		tree_matrix_nt[Node][7] = SplitValue;				// Splitting value

		if (OneSplit->NCombinations == 1)
		{
			int OneSplitVariable = OneSplit->OneSplitVariable;

			if (ncat[OneSplitVariable] > 1)
			{
				int* goright = (int *) malloc(ncat[OneSplitVariable]*sizeof(int));
				unpack(SplitValue, ncat[OneSplitVariable], goright);

				for (i = 0; i<node_n; i++)
				{
					if (goright[(int) dataX_matrix[OneSplitVariable][useObs[i]] -1] == 1)
					{
						useObsRight[RightSize] = useObs[i];
						RightSize ++;

					}else{
						useObsLeft[LeftSize] = useObs[i];
						LeftSize ++;
					}
				}

				free(goright);
			}else{
				for (i = 0; i<node_n; i++)
				{
					if (dataX_matrix[OneSplitVariable][useObs[i]] <= SplitValue)
					{
						useObsLeft[LeftSize] = useObs[i];
						LeftSize ++;
					}else{
						useObsRight[RightSize] = useObs[i];
						RightSize ++;
					}
				}
			}

			tree_matrix_nt[Node][8] = OneSplitVariable + 1;
			tree_matrix_nt[Node][8+combsplit] = NAN;

			for (j = 1; j < combsplit; j++)		// Splitting variable(s) and loading(s)
			{
				tree_matrix_nt[Node][8+j] = NAN;
				tree_matrix_nt[Node][8+combsplit+j] = NAN;
			}
		}else{

			double xcomb;
			int tempobs;

			for (i=0; i<node_n; i++)
			{
				tempobs = useObs[i];
				xcomb = 0;

				for (j=0; j< OneSplit->NCombinations; j++)
				{
					xcomb += dataX_matrix[OneSplit->SplitVariables[j]][tempobs] * OneSplit->Loadings[j];
				}

				if (xcomb <= SplitValue)
				{
					useObsLeft[LeftSize] = tempobs;
					LeftSize ++;
				}else{
					useObsRight[RightSize] = tempobs;
					RightSize ++;
				}
			}

			for (j = 0; j < OneSplit->NCombinations; j++)		// Splitting variable(s) and loading(s)
			{
				tree_matrix_nt[Node][8+j] = OneSplit->SplitVariables[j] + 1;
				tree_matrix_nt[Node][8+combsplit+j] = OneSplit->Loadings[j];
			}

			for (j = OneSplit->NCombinations; j < combsplit; j++)
			{
				tree_matrix_nt[Node][8+j] = NAN;
				tree_matrix_nt[Node][8+combsplit+j] = NAN;
			}

		}

		// release memory for the split information

		if (OneSplit->NCombinations > 1)
		{
			free(OneSplit->SplitVariables);
			free(OneSplit->Loadings);
		}

		// just be careful here...
		if (LeftSize == 0 || RightSize == 0)
		{
			if (myPara->summary >= 2) Rprintf("something wrong here... got an empty node... terminating this node...");

			free(useObsLeft);
			free(useObsRight);

			if (OneSplit->MutingUpdate)
				free(OneSplit->NewUseVariable);

			if (OneSplit->ProtectUpdate)
				free(OneSplit->NewProtectVariable);

			free(OneSplit);

			goto TerminateThisNode;
		}

		tree_matrix_nt[nextleft] = (double *) malloc( (8 + 2*combsplit + nclass) * sizeof(double));
		if (tree_matrix_nt[nextleft] == NULL) error("Unable to malloc for tree matrix");

		tree_matrix_nt[nextright] = (double *) malloc( (8 + 2*combsplit + nclass) * sizeof(double));
		if (tree_matrix_nt[nextright] == NULL) error("Unable to malloc for tree matrix");

		// fit left daughter node
		Split_A_Node_classification(nextleft, useObsLeft, dataX_matrix, dataY_vector, tree_matrix_nt, myPara, ncat, subjectweight, variableweight,
									OneSplit->MutingUpdate ? OneSplit->NewUseVariable : usevariable,
									OneSplit->ProtectUpdate ? OneSplit->NewProtectVariable : protectvariable, LeftSize,
									OneSplit->MutingUpdate ? OneSplit->newp : node_p);
		free(useObsLeft);

		// fit right daughter node
		Split_A_Node_classification(nextright, useObsRight, dataX_matrix, dataY_vector, tree_matrix_nt, myPara, ncat, subjectweight, variableweight,
									OneSplit->MutingUpdate ? OneSplit->NewUseVariable : usevariable,
									OneSplit->ProtectUpdate ? OneSplit->NewProtectVariable : protectvariable, RightSize,
									OneSplit->MutingUpdate ? OneSplit->newp : node_p);
		free(useObsRight);

		if (OneSplit->MutingUpdate)
			free(OneSplit->NewUseVariable);

		if (OneSplit->ProtectUpdate)
			free(OneSplit->NewProtectVariable);

		free(OneSplit);
	}
}

SplitRule* Find_A_Split_classification(int* useObs,
									   double** dataX_matrix,
									   int* dataY_vector,
									   PARAMETERS* myPara,
									   int* ncat,
									   double* subjectweight,
									   double* variableweight,
									   int* usevariable,
									   int* protectvariable,
									   int node_n,
									   int node_p,
									   int root)
{
	// Get parameters
	int nmin = myPara->nmin;
	int mtry = myPara->mtry;
	int nclass = myPara->nclass;
	int split_gen = myPara->split_gen;
	int nspliteach = myPara->nspliteach;
	// int select_method = myPara->select_method;
	int reinforcement = myPara->reinforcement;
	int use_sub_weight = myPara->use_sub_weight;
	int use_var_weight = myPara->use_var_weight;

	SplitRule *OneSplit = malloc(sizeof(SplitRule)); 	// Create a split rule
	OneSplit->NCombinations = 0; 						// When NCombinations == 0, no split is found
	OneSplit->MutingUpdate = 0;
	OneSplit->ProtectUpdate = 0;

	// Check Identical Y
	if (CheckIdentical_i(dataY_vector, useObs, node_n))
		return(OneSplit);

	int i;
	int j;
	int bestVar = 0;
	double bestScore = -1;
	double bestValue = 0;
	int use_var;

	if (reinforcement & (node_n >= 2*nmin) & (node_p > 1))
	{

		// setup parameters
		PARAMETERS *myPara_embed = malloc(sizeof(PARAMETERS));
		myPara_embed->data_n = myPara->data_n;
		myPara_embed->dataX_p = myPara->dataX_p;
		myPara_embed->summary = 0;
		myPara_embed->useCores = myPara->useCores;
		myPara_embed->ntrees = myPara->ntrees_embed;
		myPara_embed->mtry = imax(node_p, imax(1, node_p*myPara->mtry_embed));
		myPara_embed->nmin = myPara->nmin_embed;
		myPara_embed->split_gen = myPara->split_gen_embed;
		myPara_embed->nspliteach = myPara->nspliteach_embed;
		myPara_embed->select_method = myPara->select_method;
		myPara_embed->nclass = myPara->nclass;
		myPara_embed->replacement = 0;
		myPara_embed->npermute = 1;
		myPara_embed->reinforcement = 0;
		myPara_embed->combsplit = 1;
		myPara_embed->importance = 1;
		myPara_embed->use_sub_weight = use_sub_weight;
		myPara_embed->use_var_weight = use_var_weight;
		myPara_embed->track_obs = 0;
		myPara_embed->resample_prob = myPara->resample_prob_embed;

		int nt;
		int ntrees = myPara_embed->ntrees;
		// get tree matrix
		int TreeMaxLength = 1 + 2*node_n;

		// fit embedded model
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

		// variable importance for embedded model
		double **AllError = (double **) malloc(ntrees * sizeof(double *));
		if (AllError == NULL) error("Unable to malloc (embedded)");

		double *VarImp = calloc(node_p, sizeof(double));
		if (VarImp == NULL) error("Unable to malloc (embedded)");

		for (nt=0; nt<ntrees; nt++)
		{
			AllError[nt] = (double *) calloc((node_p+1), sizeof(double));
			if (AllError[nt] == NULL) error("Unable to malloc (embedded)");
		}
		int** obs_in_trees = NULL;

		// fit
		Fit_Trees_classification(dataX_matrix, dataY_vector, tree_matrix, AllError, VarImp, obs_in_trees, myPara_embed, ncat, subjectweight, variableweight, useObs, usevariable, protectvariable, node_n, node_p);

		//print_d_mat_t(AllError, ntrees, node_p+1);

		// remove unused objects
		for (nt = 0; nt<ntrees; nt++)
		{
			for (i = 0; i < TreeMaxLength; i++)
				if (tree_matrix[nt][i] != NULL)
					free(tree_matrix[nt][i]);

			free(tree_matrix[nt]);

			free(AllError[nt]);
		}
		free(tree_matrix);
		free(AllError);
		free(myPara_embed);

		int* index = (int *) malloc(node_p * sizeof(int));
		for (j = 0; j<node_p; j++)
			index[j] = usevariable[j];

		// print_i_d_vec(index, VarImp, node_p);

		rsort_with_index(VarImp, index, node_p);

		// Rprintf("get embeded importance: \n");
		// print_i_d_vec(index, VarImp, node_p);

		int combsplit = imin(node_p, myPara->combsplit);

		if (ISNA(VarImp[node_p-1]))
		{
			// Rprintf("variable importance missing, go to single split \n");
			free(index);
			free(VarImp);
			goto DoASingleSplit; // nothing good
		}

		if (VarImp[node_p-1] <= 0)
		{
			// Rprintf("best importance is negative, go to single split \n");
			free(index);
			free(VarImp);
			goto DoASingleSplit; // nothing good
		}

		// otherwise, I look at the first combsplit variables with positive VI
		// if there are catigorical variables within them, random sample one variable from combsplit, if it is catigorical, use as the splitting variable
		// if it is a continuous variable, use all the continuous ones in combsplit to create a linear combination

		int real_comb = 0;

		if (ncat[index[node_p-1]] > 1)
		{
			real_comb = 1;
		}else{
			for (j=0; j< combsplit; j++)
			{
				if ((ncat[index[node_p-1-j]] == 1) & (VarImp[node_p-1-j]/VarImp[node_p-1] >= myPara->combsplit_th))
				{
					real_comb ++;
				}else{
					break;
				}
			}
		}

		//Rprintf("got combination %i \n", real_comb);

		OneSplit->ProtectUpdate = 1;
		OneSplit->NewProtectVariable = (int *) malloc(myPara->dataX_p * sizeof(int));
		for (j=0; j< myPara->dataX_p; j++) OneSplit->NewProtectVariable[j] = protectvariable[j];

		if (root)
		{
			for (j = 0; j < myPara->protectVar; j++)
			{
				if ((VarImp[node_p-1-j] > 0) & (j < node_p))
				{
					OneSplit->NewProtectVariable[index[node_p-1-j]] = 1;
				}else{
					break;
				}
			}
		}

		if ((real_comb == 1) | (nclass > 2) ) // use one variable split
		{
			use_var = index[node_p-1];

			if (ncat[use_var] > 1)
			{
				OneSplit_Cat_classification(&bestValue, &bestScore, dataX_matrix[use_var], dataY_vector, subjectweight, useObs, use_sub_weight, ncat[use_var], node_n, nclass, split_gen == 4 ? 4 : 3, nspliteach, nmin);
			}else{
				OneSplit_Cont_classification(&bestValue, &bestScore, dataX_matrix[use_var], dataY_vector, subjectweight, useObs, use_sub_weight, node_n, nclass, split_gen == 4 ? 4 : 3, nspliteach, nmin);
			}

			OneSplit->NCombinations = 1;
			OneSplit->OneSplitVariable = use_var;
			OneSplit->SplitValue = bestValue;
			OneSplit->NewProtectVariable[use_var] = 1;

			//Rprintf("One splitting variable %i, best split at %f, with score %f \n", use_var, bestValue, bestScore);
		}else{
			// get variables and loadings in the combination

			OneSplit->NCombinations = real_comb;
			OneSplit->SplitVariables = (int *) malloc(real_comb*sizeof(int));
			OneSplit->Loadings = (double *) malloc(real_comb*sizeof(double));

			double* xcomb = (double *) malloc(myPara->data_n * sizeof(double));
			double xsum0;
			double xsum1;
			int count1;
			int tempobs;
			int tempvar;

			// determine sign of loading
			for (j = 0; j < real_comb; j++)
			{
				tempvar = index[node_p-1-j];
				xsum1=0;
				xsum0=0;
				count1 = 0;

				for (i=0; i<node_n; i++)
				{
					tempobs = useObs[i];
					if (dataY_vector[tempobs] == 1)
					{
						xsum1 += dataX_matrix[tempvar][tempobs];
						count1 ++;
					}else{
						xsum0 += dataX_matrix[tempvar][tempobs];
					}
				}

				OneSplit->SplitVariables[j] = tempvar;
				OneSplit->Loadings[j] = sqrt(VarImp[node_p-1-j])*sgn_rand( xsum1/count1 - xsum0/(node_n-count1) );
			}

			// get loading vectors
			for (j = real_comb-1; j > 0; j--)
			{
				OneSplit->Loadings[j] /= OneSplit->Loadings[0];
			}
			OneSplit->Loadings[0] = 1;

/* 			for (j=0; j<real_comb; j++)
			{
				Rprintf("linear comb %i variable %i, loading %f \n", j+1, OneSplit->SplitVariables[j] +1, OneSplit->Loadings[j]);
			} */

			for (i=0; i<node_n; i++)
			{
				tempobs = useObs[i];
				xcomb[tempobs] = 0;
				// Rprintf("subject %i ", tempobs);

				for (j=0; j< real_comb; j++)
				{
					//Rprintf("variable %i, value %f, loading %f  ", OneSplit->SplitVariables[j]+1, dataX_matrix[OneSplit->SplitVariables[j]][tempobs], OneSplit->Loadings[j]);
					xcomb[tempobs] += dataX_matrix[OneSplit->SplitVariables[j]][tempobs] * OneSplit->Loadings[j];

				}
				//Rprintf("\n");
				//Rprintf("subject %i combination %f \n", tempobs, xcomb[tempobs]);
			}

			OneSplit_Cont_classification(&bestValue, &bestScore, xcomb, dataY_vector, subjectweight, useObs, use_sub_weight, node_n, nclass, split_gen == 4 ? 4 : 3, nspliteach, nmin);

			//Rprintf("found best split at %f, with score %f \n", bestValue, bestScore);

			// update splitting rule
			OneSplit->SplitValue = bestValue;
			// update protected variables
			for (j=0; j< real_comb; j++)
				OneSplit->NewProtectVariable[ OneSplit->SplitVariables[j] ] = 1;

			free(xcomb);
		}

		// use index to indicate muted variables, then copy unmuted to the splitting rule

		use_var = 0; // use this to denote number of muted
		bestVar = 0; // use this to denote number of used

		if (myPara->muting == -1)
			use_var = imin(node_p-real_comb, node_p*myPara->muting_percent);

		if (myPara->muting >= 1)
			use_var = imin(node_p-real_comb, myPara->muting);

		//Rprintf("Muting method is %i will mute %i variables", myPara->muting, use_var);

		if (use_var > 0)
		{
			for (j=0; j< use_var; j++)
			{
				if (OneSplit->NewProtectVariable[index[j]] == 0)
					index[j] = -1;
			}

			//Rprintf("Variable muting information \n");
			//print_i_d_vec(index, VarImp, node_p);

			OneSplit->MutingUpdate = 1;

			for (j=0; j< node_p; j++)
				if (index[j] >=0)
					bestVar++;

			OneSplit->NewUseVariable = (int *) malloc(bestVar*sizeof(int));
			OneSplit->newp = bestVar;

			use_var = 0; // as index mover

			for (j=0; j< node_p; j++)
			{
				if (index[j] >=0)
				{
					OneSplit->NewUseVariable[use_var] = index[j];
					use_var ++;
				}
			}
		}

		free(index);
		free(VarImp);

	}else{
		// start to find splits

DoASingleSplit:;

		int totalp = node_p;
		int randVar;
		double tempValue;
		double tempScore;

		int x_cat;
		int real_mtry = imin(mtry, totalp);
		double* allVarWeight = NULL;
		int* allVar = NULL;

		if (use_var_weight)
		{
			allVarWeight = (double *) malloc(node_p * sizeof(double));
			for (j = 0; j<node_p; j++)
				allVarWeight[j] = variableweight[usevariable[j]];

			standardize(allVarWeight, node_p);
		}else{
			allVar = (int *) malloc(node_p * sizeof(int));
			for (j = 0; j<node_p; j++)
				allVar[j] = usevariable[j];
		}

		while( real_mtry > 0 )	// randomly sample a variable
		{
			if (use_var_weight)
			{
				use_var = usevariable[sample(allVarWeight, node_p)];
			}else{
				randVar = random_in_range(0, totalp);
				use_var = allVar[randVar];	// sample a splitting variable
				allVar[randVar] = allVar[totalp-1];
				totalp --;
			}

			real_mtry --;

			if(CheckIdentical_d(dataX_matrix[use_var], useObs, node_n) == 0)
			{

				x_cat = ncat[use_var];
				// Rprintf("get variable %i \n", use_var);

				if (x_cat > 1) // for a categorical variable
				{
					// Rprintf("Variable %i is a categorical variable \n", use_var+1);
					OneSplit_Cat_classification(&tempValue, &tempScore, dataX_matrix[use_var], dataY_vector, subjectweight, useObs, use_sub_weight, x_cat, node_n, nclass, split_gen, nspliteach, nmin);
					// Rprintf("Variable %i is a categorical variable, get score is %f \n", use_var+1, tempScore);
				}else{
					// Rprintf("Variable %i is a continuous variable \n", use_var+1);
					OneSplit_Cont_classification(&tempValue, &tempScore, dataX_matrix[use_var], dataY_vector, subjectweight, useObs, use_sub_weight, node_n, nclass, split_gen, nspliteach, nmin);
					// Rprintf("Variable %i is a continuous variable, get score is %f \n", use_var+1, tempScore);
				}

				if (tempScore > bestScore)
				{
					bestVar = use_var;
					bestValue = tempValue;
					bestScore = tempScore;
				}
			}
		}

		if (bestScore > 0)
		{
			OneSplit->NCombinations = 1; 				// regular split always with 1 splitting variable
			OneSplit->OneSplitVariable = bestVar;		// for 1 split, loading is not needed
			OneSplit->SplitValue = bestValue;
		}

		if (use_var_weight)
		{
			free(allVarWeight);
		}else{
			free(allVar);
		}
	}

	return(OneSplit);
}

void OneSplit_Cont_classification(double *cutValue, double* score, double* x, int* y, double* weights, int* useObs, int use_weight, int n, int nclass, int split_gen, int nspliteach, int nmin)
{
	*cutValue = NAN;
	*score = -1;

	int i;
	double a_random_cut;
	double a_random_score = -1;

	if (split_gen == 1) // random
	{
		if (use_weight)
		{
			for (i=0; i<nspliteach; i++)
			{
				a_random_cut = x[useObs[random_in_range(0, n)]];
				a_random_score = score_at_cut_cla_w(x, y, weights, useObs, n, nclass, a_random_cut);

				if (a_random_score > *score)
				{
					*cutValue = a_random_cut;
					*score = a_random_score;
				}
			}
		}else{
			for (i=0; i<nspliteach; i++)
			{
				a_random_cut = x[useObs[random_in_range(0, n)]];
				a_random_score = score_at_cut_cla(x, y, useObs, n, nclass, a_random_cut);

				if (a_random_score > *score)
				{
					*cutValue = a_random_cut;
					*score = a_random_score;
				}
			}
		}
		return;
	}

	if (split_gen == 2) // uniform
	{
		double xmin = x[0];
		double xmax = x[0];

		get_max_min(&xmin, &xmax, x, useObs, n);

		if (use_weight)
		{
			for (i=0; i<nspliteach; i++)
			{
				a_random_cut = xmin + (xmax-xmin)*unif_rand();
				a_random_score = score_at_cut_cla_w(x, y, weights, useObs, n, nclass, a_random_cut);

				if (a_random_score > *score)
				{
					*cutValue = a_random_cut;
					*score = a_random_score;
				}
			}
		}else{
			for (i=0; i<nspliteach; i++)
			{
				a_random_cut = xmin + (xmax-xmin)*unif_rand();
				a_random_score = score_at_cut_cla(x, y, useObs, n, nclass, a_random_cut);

				if (a_random_score > *score)
				{
					*cutValue = a_random_cut;
					*score = a_random_score;
				}
			}
		}
		return;
	}

	if (split_gen == 3) // rank
	{
		int a_random_rank;

		if (use_weight)
		{
			// copy and sort
			struct_xcw* copy_xyw = (struct_xcw*) malloc(n * sizeof(struct_xcw));
			for (i =0; i< n; i++)
			{
				copy_xyw[i].x = x[useObs[i]];
				copy_xyw[i].y = y[useObs[i]];
				copy_xyw[i].w = weights[useObs[i]];
			}

			qsort(copy_xyw, n, sizeof(struct_xcw), compare_struct_xcw);

			for (i=0; i<nspliteach; i++)
			{
				a_random_rank = random_in_range((int) nmin/2, n - ((int) nmin/2) + 1); // get a random rank for split

				// if ties with boundary
				while (copy_xyw[a_random_rank].x == copy_xyw[0].x) a_random_rank++;
				while (copy_xyw[a_random_rank-1].x == copy_xyw[n-1].x) a_random_rank--;

				if (unif_rand() > 0.5) // while in the middle of a sequence of ties, either move up or move down
				{
					while (copy_xyw[a_random_rank].x == copy_xyw[a_random_rank-1].x) a_random_rank++;
				}else{
					while (copy_xyw[a_random_rank].x == copy_xyw[a_random_rank-1].x) a_random_rank--;
				}

				a_random_score = score_at_rank_cla_w(copy_xyw, n, nclass, a_random_rank);

				if (a_random_score > *score)
				{
					*cutValue = (copy_xyw[a_random_rank-1].x + copy_xyw[a_random_rank].x)/2;
					*score = a_random_score;
				}
			}

			free(copy_xyw);
		}else{
			struct_xc* copy_xy = (struct_xc*) malloc(n * sizeof(struct_xc));
			for (i =0; i< n; i++)
			{
				copy_xy[i].x = x[useObs[i]];
				copy_xy[i].y = y[useObs[i]];
			}
			qsort(copy_xy, n, sizeof(struct_xc), compare_struct_xc);

			for (i=0; i<nspliteach; i++)
			{
				a_random_rank = random_in_range((int) nmin/2, n - ((int) nmin/2) + 1); // get a random rank for split

				// if ties with boundary
				while (copy_xy[a_random_rank].x == copy_xy[0].x) a_random_rank++;
				while (copy_xy[a_random_rank-1].x == copy_xy[n-1].x) a_random_rank--;

				if (unif_rand() > 0.5) // while in the middle of a sequence of ties, either move up or move down
				{
					while (copy_xy[a_random_rank].x == copy_xy[a_random_rank-1].x) a_random_rank++;
				}else{
					while (copy_xy[a_random_rank].x == copy_xy[a_random_rank-1].x) a_random_rank--;
				}

				// get cut and score
				a_random_score = score_at_rank_cla(copy_xy, n, nclass, a_random_rank);

				if (a_random_score > *score)
				{
					*cutValue = (copy_xy[a_random_rank-1].x + copy_xy[a_random_rank].x)/2;
					*score = a_random_score;
				}
			}

			free(copy_xy);
		}
		return;
	}

	if (split_gen == 4) // best
	{
		if (use_weight)
		{
			struct_xcw* copy_xyw = (struct_xcw*) malloc(n * sizeof(struct_xcw));
			for (i =0; i< n; i++)
			{
				copy_xyw[i].x = x[useObs[i]];
				copy_xyw[i].y = y[useObs[i]];
				copy_xyw[i].w = weights[useObs[i]];
			}

			qsort(copy_xyw, n, sizeof(struct_xcw), compare_struct_xcw);

			score_best_cla_w(copy_xyw, n, nclass, cutValue, score);

			free(copy_xyw);
		}else{

/* 			double* xcopy = (double*) malloc(n*sizeof(double));
			int* index = (int*) malloc(n*sizeof(int));
			for (i=0; i< n; i++)
			{
				xcopy[i] = x[useObs[i]];
				index[i] = useObs[i];
			}
			rsort_with_index(xcopy, index, n);
			temp_score(xcopy, y, index, n, nclass, cutValue, score);
			free(xcopy);
			free(index); */

			struct_xc* copy_xy = (struct_xc*) malloc(n * sizeof(struct_xc));
			for (i =0; i< n; i++)
			{
				copy_xy[i].x = x[useObs[i]];
				copy_xy[i].y = y[useObs[i]];
			}
			qsort(copy_xy, n, sizeof(struct_xc), compare_struct_xc);
			score_best_cla(copy_xy, n, nclass, cutValue, score);
			free(copy_xy);
		}
	}
}

void OneSplit_Cat_classification(double *cutValue, double *score, double* x, int* y, double* weights, int* useObs, int use_weight, int x_cat, int n, int nclass, int split_gen, int nspliteach, int nmin)
{
	*cutValue = NAN;
	*score = -1;

	int i;
	int k;
	int temp_cat;
	int true_x_cat = 0;

	// summarize this categorical variable
	struct_cat_cla* cat_count = (struct_cat_cla*) malloc(x_cat * sizeof(struct_cat_cla));

	for (i=0; i< x_cat; i++)
	{
		cat_count[i].cat = i;
		cat_count[i].cla = calloc(nclass, sizeof(double));
		cat_count[i].wsum = 0;
	}

	if (use_weight)
	{
		for (i=0; i<n; i++)
		{
			cat_count[(int) x[useObs[i]] -1].cla[y[useObs[i]]] += weights[useObs[i]];
		}
	}else{
		for (i=0; i<n; i++)
		{
			cat_count[(int) x[useObs[i]] -1].cla[y[useObs[i]]] ++;
		}
	}

	for (i=0; i<x_cat; i++)
	{
		for (k=0; k<nclass; k++)
			cat_count[i].wsum += cat_count[i].cla[k];
	}

	// put the nonzero ones to the front
	true_x_cat = x_cat;
	for (i =0; i < true_x_cat; i++)
	{
		if (cat_count[i].wsum <= 0)
		{
			swap_cat_cla(&cat_count[i], &cat_count[true_x_cat-1]);
			true_x_cat--;
			i--;
		}
	}

	if (true_x_cat <= 1)
		goto NothingToFind;

	int* goright = (int *) malloc(x_cat*sizeof(int));
	double* YLeft = (double *) malloc(nclass*sizeof(double));
	double* YRight = (double *) malloc(nclass*sizeof(double));
	double tempLeft;
	double tempRight;
	double leftweight;
	double rightweight;
	double tempscore;
	int j;
	int rightcount;

	if (nclass == 2) // for two class, we can order the samples
	{
		// to sort the categories
		// this is for a little trick to get around the categories that do not exist at the current node (but some should be)
		// I will only select the nonzero categories from the lowest rank to go right
		if (unif_rand() > 0.5)
		{
			qsort(cat_count, true_x_cat, sizeof(struct_cat_cla), compare_struct_cat_cla_two);
		}else{
			qsort(cat_count, true_x_cat, sizeof(struct_cat_cla), compare_struct_cat_cla_two_rev);
		}

		if (split_gen == 1 || split_gen == 2 || split_gen == 3)
		{
			for (j =0; j<nspliteach; j++)
			{
				memset(goright, 0, x_cat*sizeof(int));
				memset(YLeft, 0, nclass*sizeof(double));
				memset(YRight, 0, nclass*sizeof(double));
				leftweight = 0;
				rightweight = 0;

				temp_cat = random_in_range(1, true_x_cat);

				// add up the smaller categories
				for (i=0; i<temp_cat; i++)
				{
					goright[cat_count[i].cat] = 1;
					for (k=0; k<nclass; k++) YRight[k] += cat_count[i].cla[k];
					rightweight += cat_count[i].wsum;
				}

				// add up the larger categories
				for (i = temp_cat; i < true_x_cat; i++)
				{
					for (k=0; k<nclass; k++) YLeft[k] += cat_count[i].cla[k];
					leftweight += cat_count[i].wsum;
				}

				// get score
				tempLeft = 0;
				tempRight = 0;
				for (i=0; i< nclass; i++)
				{
					tempLeft += YLeft[i]*YLeft[i];
					tempRight += YRight[i]*YRight[i];
				}

				tempscore = tempLeft/leftweight + tempRight/rightweight;

				if (tempscore > *score)
				{
					*score = tempscore;
					*cutValue = pack(x_cat, goright);
				}
			}
		}else{ // best split

			//Rprintf("Number of categories is %i \n", true_x_cat);
			memset(goright, 0, x_cat*sizeof(int));
			memset(YLeft, 0, nclass*sizeof(double));
			memset(YRight, 0, nclass*sizeof(double));
			tempLeft = 0;
			tempRight = 0;
			leftweight = 0;
			rightweight = 0;

			for (k=0; k<nclass; k++) YRight[k] += cat_count[0].cla[k];
			rightweight += cat_count[0].wsum;
			goright[cat_count[0].cat] = 1;

			rightcount = 1;

			for (i = rightcount; i<true_x_cat; i++)
			{
				for (k=0; k<nclass; k++) YLeft[k] += cat_count[i].cla[k];
				leftweight += cat_count[i].wsum;
			}

			for (k=0; k< nclass; k++)
			{
				tempLeft += YLeft[k]*YLeft[k];
				tempRight += YRight[k]*YRight[k];
			}

			tempscore = tempLeft/leftweight + tempRight/rightweight;
			//Rprintf("N cat go right %i, tempLeft is %f, leftweight is %f, tempRight is %f, rightweight is %f, tempscore is %f \n", rightcount, tempLeft, leftweight, tempRight, rightweight, tempscore);
			if (tempscore > *score)
			{
				*cutValue = pack(x_cat, goright);
				*score = tempscore;
			}

			// go through the rest cut points

			for (i = rightcount; i<true_x_cat-1; i++)
			{
				for (k=0; k<nclass; k++)
				{
					YRight[k] += cat_count[i].cla[k];
					YLeft[k] -= cat_count[i].cla[k];
				}

				rightweight += cat_count[i].wsum;
				leftweight -= cat_count[i].wsum;

				tempLeft = 0;
				tempRight = 0;
				goright[cat_count[i].cat] = 1;

				for (k=0; k< nclass; k++)
				{
					tempLeft += YLeft[k]*YLeft[k];
					tempRight += YRight[k]*YRight[k];
				}

				tempscore = tempLeft/leftweight + tempRight/rightweight;

				if (tempscore > *score)
				{
					*cutValue = pack(x_cat, goright);
					*score = tempscore;
				}

				//Rprintf("N cat go right %i, tempLeft is %f, leftweight is %f, tempRight is %f, rightweight is %f, tempscore is %f \n", i+1, tempLeft, leftweight, tempRight, rightweight, tempscore);
			}
		}
	}else{
		int* randomRight = (int *) malloc(true_x_cat*sizeof(int));

		if (split_gen == 1 || split_gen == 2 || split_gen == 3)
		{
			if (true_x_cat == 2) nspliteach = 1;
			if (nspliteach > pow(2, true_x_cat-1)) goto BestSplittingsForMultipleClass;

		RandomSplittingsForMultipleClass:
			//Rprintf("random splitting rule for more than two classes \n");

			for (j =0; j<nspliteach; j++)
			{
				memset(goright, 0, x_cat*sizeof(int));
				memset(randomRight, 0, true_x_cat*sizeof(int));
				memset(YLeft, 0, nclass*sizeof(double));
				memset(YRight, 0, nclass*sizeof(double));
				leftweight = 0;
				rightweight = 0;

				// get a random splitting rule
				for (i=0; i<true_x_cat; i++) randomRight[i] = 1;
				memset(randomRight, 0, random_in_range(1, true_x_cat)*sizeof(int));
				permute(randomRight, true_x_cat);

				for (i = 0; i< true_x_cat; i++)
				{
					if (randomRight[i] == 1)
					{
						goright[cat_count[i].cat] = 1;
						for (k=0; k<nclass; k++) YRight[k] += cat_count[i].cla[k];
						rightweight += cat_count[i].wsum;
					}else{
						for (k=0; k<nclass; k++) YLeft[k] += cat_count[i].cla[k];
						leftweight += cat_count[i].wsum;
					}
				}

				tempLeft = 0;
				tempRight = 0;

				for (k=0; k< nclass; k++)
				{
					tempLeft += YLeft[k]*YLeft[k];
					tempRight += YRight[k]*YRight[k];
				}

				tempscore = tempLeft/leftweight + tempRight/rightweight;

				//Rprintf("Random split %i, tempLeft is %f, leftweight is %f, tempRight is %f, rightweight is %f, tempscore is %f \n", j+1, tempLeft, leftweight, tempRight, rightweight, tempscore);

				if (tempscore > *score)
				{
					*cutValue = pack(x_cat, goright);
					*score = tempscore;
				}
			}
		}else{

			if (true_x_cat > 10)
			{
				nspliteach = 1024;
				goto RandomSplittingsForMultipleClass;
			}

		BestSplittingsForMultipleClass:

			//Rprintf("best splitting rule for more than two classes \n");

			memset(goright, 0, x_cat*sizeof(int));
			memset(randomRight, 0, true_x_cat*sizeof(int));
			memset(YLeft, 0, nclass*sizeof(double));
			memset(YRight, 0, nclass*sizeof(double));
			leftweight = 0;
			rightweight = 0;
			nspliteach = pow(2, true_x_cat-1);

			// initial everything to the left node
			for (i=0; i < true_x_cat; i++)
			{
				for (k = 0; k< nclass; k++) YLeft[k] += cat_count[i].cla[k];
				leftweight += cat_count[i].wsum;
			}

			// use randomRight as the movetoright indicator
			// keep adding 1 to the first element of randomRight, and make binary changes to all other elements
			// update left and right information and goright correspondingly and calculate score

			for (j = 0; j < nspliteach; j++)
			{
				randomRight[0]++;

				if (randomRight[0] == 1)
				{
					//Rprintf("Adding 0 \n");
					goright[cat_count[0].cat] = 1;
 					for (k = 0; k< nclass; k++)
					{
						YRight[k] += cat_count[0].cla[k];
						YLeft[k] -= cat_count[0].cla[k];
					}
					rightweight += cat_count[0].wsum;
					leftweight -= cat_count[0].wsum;
				}

				for (i = 0; i< true_x_cat-1; i++)
				{
					if (randomRight[i] == 2)
					{
						//Rprintf("minus %i \n", i);
						goright[cat_count[i].cat] = 0;
						for (k = 0; k< nclass; k++)
						{
							YRight[k] -= cat_count[i].cla[k];
							YLeft[k] += cat_count[i].cla[k];
						}
						rightweight -= cat_count[i].wsum;
						leftweight += cat_count[i].wsum;

						randomRight[i] = 0;
						randomRight[i+1]++;

						if (randomRight[i+1] == 1)
						{
							//Rprintf("adding %i \n", i+1);
							goright[cat_count[i+1].cat] = 1;
							for (k = 0; k< nclass; k++)
							{
								YRight[k] += cat_count[i+1].cla[k];
								YLeft[k] -= cat_count[i+1].cla[k];
							}
							rightweight += cat_count[i+1].wsum;
							leftweight -= cat_count[i+1].wsum;
						}
					}
				}

				tempLeft = 0;
				tempRight = 0;

				for (k=0; k< nclass; k++)
				{
					tempLeft += YLeft[k]*YLeft[k];
					tempRight += YRight[k]*YRight[k];
				}

				tempscore = tempLeft/leftweight + tempRight/rightweight;

				//Rprintf("Random split %i, tempLeft is %f, leftweight is %f, tempRight is %f, rightweight is %f, tempscore is %f \n", j+1, tempLeft, leftweight, tempRight, rightweight, tempscore);

				if (tempscore > *score)
				{
					*cutValue = pack(x_cat, goright);
					*score = tempscore;
				}

				//print_i_vec_t(goright, x_cat);
			}
		}
	}

	free(YLeft);
	free(YRight);
	free(goright);

NothingToFind:

	for (i=0; i< x_cat; i++)
		free(cat_count[i].cla);
	free(cat_count);
}

// prediction functions

void predict_cla(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int combsplit, int* ncat, int* oobObs, int* Ypred, int oobN)
{
	int i;

/*  FILE * Output;
	Output = fopen("error.txt", "a");
	fprintf(Output, " --- at this node %i , n subject is %i --- \n", Node, oobN);
	for (i=0; i< oobN; i++)
	{
		fprintf(Output, " Yind %i,  oobObs[Yind[i]] %i \n", Yind[i], oobObs[Yind[i]]);
	}
	fclose(Output);  */

	if (tree_matrix_nt[Node][0] == 2)
	{
		for (i=0; i<oobN; i++)
			Ypred[Yind[i]] = (int) tree_matrix_nt[Node][3];
	}else{

		int* leftNode = (int *) malloc(oobN * sizeof(int));
		int* rightNode = (int *) malloc(oobN * sizeof(int));
		int leftCount = 0;
		int rightCount = 0;
		int splitVar;
		double splitPoint = tree_matrix_nt[Node][7];

		if (tree_matrix_nt[Node][6] == 1) // one variable split
		{
			splitVar = (int) tree_matrix_nt[Node][8] - 1;

			if (ncat[splitVar] > 1)	// categorical variable
			{

				int* goright = (int *) malloc(ncat[splitVar]*sizeof(int));

				unpack(splitPoint, ncat[splitVar], goright);

				for (i =0; i<oobN; i++)
				{
					if (goright[(int) dataX_matrix[splitVar][oobObs[Yind[i]]] -1] == 1)
					{
						rightNode[rightCount] = Yind[i];
						rightCount++;
					}else{
						leftNode[leftCount] = Yind[i];
						leftCount++;
					}
				}
				free(goright);
			}else{
				for (i=0; i<oobN; i++)
				{
					if (dataX_matrix[splitVar][oobObs[Yind[i]]] > splitPoint)
					{
						rightNode[rightCount] = Yind[i];
						rightCount++;
					}else{
						leftNode[leftCount] = Yind[i];
						leftCount++;
					}
				}
			}
		}else{

			int ncomb = tree_matrix_nt[Node][6];
			int k;
			double xcomb;
			int* xvar = (int *) malloc(ncomb * sizeof(int));
			double* loading = (double *) malloc(ncomb * sizeof(double));

			for (k = 0; k<ncomb; k++)
			{
				xvar[k] = (int) tree_matrix_nt[Node][8+k] - 1;
				loading[k] = tree_matrix_nt[Node][8+combsplit+k];
			}

			for (i=0; i<oobN; i++)
			{
				xcomb = 0;

				for (k=0; k< ncomb; k++)
				{
					xcomb += dataX_matrix[xvar[k]][oobObs[Yind[i]]] * loading[k];
				}

				if (xcomb > splitPoint)
				{
					rightNode[rightCount] = Yind[i];
					rightCount++;
				}else{
					leftNode[leftCount] = Yind[i];
					leftCount++;
				}
			}
			free(xvar);
			free(loading);

		}
		predict_cla((int)tree_matrix_nt[Node][4], leftNode, dataX_matrix, tree_matrix_nt, combsplit, ncat, oobObs, Ypred, leftCount);
		free(leftNode);

		predict_cla((int)tree_matrix_nt[Node][5], rightNode, dataX_matrix, tree_matrix_nt, combsplit, ncat, oobObs, Ypred, rightCount);
		free(rightNode);
	}
}


void predict_cla_all(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int nclass, int combsplit, int* ncat, int* oobObs, double** Ypred, int oobN)
{
	int i;

/*  FILE * Output;
	Output = fopen("error.txt", "a");
	fprintf(Output, " --- at this node %i , n subject is %i --- \n", Node, oobN);
	for (i=0; i< oobN; i++)
	{
		fprintf(Output, " Yind %i,  oobObs[Yind[i]] %i \n", Yind[i], oobObs[Yind[i]]);
	}
	fclose(Output);  */

	if (tree_matrix_nt[Node][0] == 2)
	{
		int PartWidth = 8 + 2*combsplit;
		int j;

 		for (i=0; i<oobN; i++)
			for (j=0; j<nclass; j++)
				Ypred[Yind[i]][j] = tree_matrix_nt[Node][PartWidth + j];

	}else{

		int* leftNode = (int *) malloc(oobN * sizeof(int));
		int* rightNode = (int *) malloc(oobN * sizeof(int));
		int leftCount = 0;
		int rightCount = 0;
		int splitVar;
		double splitPoint = tree_matrix_nt[Node][7];

		if (tree_matrix_nt[Node][6] == 1) // one variable split
		{
			splitVar = (int) tree_matrix_nt[Node][8] - 1;

			if (ncat[splitVar] > 1)	// categorical variable
			{

				int* goright = (int *) malloc(ncat[splitVar]*sizeof(int));

				unpack(splitPoint, ncat[splitVar], goright);

				for (i =0; i<oobN; i++)
				{
					if (goright[(int) dataX_matrix[splitVar][oobObs[Yind[i]]] -1] == 1)
					{
						rightNode[rightCount] = Yind[i];
						rightCount++;
					}else{
						leftNode[leftCount] = Yind[i];
						leftCount++;
					}
				}
				free(goright);
			}else{
				for (i=0; i<oobN; i++)
				{
					if (dataX_matrix[splitVar][oobObs[Yind[i]]] > splitPoint)
					{
						rightNode[rightCount] = Yind[i];
						rightCount++;
					}else{
						leftNode[leftCount] = Yind[i];
						leftCount++;
					}
				}
			}
		}else{

			int ncomb = tree_matrix_nt[Node][6];
			int k;
			double xcomb;
			int* xvar = (int *) malloc(ncomb * sizeof(int));
			double* loading = (double *) malloc(ncomb * sizeof(double));

			for (k = 0; k<ncomb; k++)
			{
				xvar[k] = (int) tree_matrix_nt[Node][8+k] - 1;
				loading[k] = tree_matrix_nt[Node][8+combsplit+k];
			}

			for (i=0; i<oobN; i++)
			{
				xcomb = 0;

				for (k=0; k< ncomb; k++)
				{
					xcomb += dataX_matrix[xvar[k]][oobObs[Yind[i]]] * loading[k];
				}

				if (xcomb > splitPoint)
				{
					rightNode[rightCount] = Yind[i];
					rightCount++;
				}else{
					leftNode[leftCount] = Yind[i];
					leftCount++;
				}
			}
			free(xvar);
			free(loading);

		}
		predict_cla_all((int)tree_matrix_nt[Node][4], leftNode, dataX_matrix, tree_matrix_nt, nclass, combsplit, ncat, oobObs, Ypred, leftCount);
		free(leftNode);

		predict_cla_all((int)tree_matrix_nt[Node][5], rightNode, dataX_matrix, tree_matrix_nt, nclass, combsplit, ncat, oobObs, Ypred, rightCount);
		free(rightNode);
	}
}
// prediction with permutation

void predict_cla_pj(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int combsplit, int* ncat, int* oobObs, int* Ypred, int oobN, int* permuteInt, int j)
{
	int i;

	if (tree_matrix_nt[Node][0] == 2)
	{
		for (i=0; i<oobN; i++)
			Ypred[Yind[i]] = (int) tree_matrix_nt[Node][3];
	}else{

		int* leftNode = (int *) malloc(oobN * sizeof(int));
		int* rightNode = (int *) malloc(oobN * sizeof(int));
		int leftCount = 0;
		int rightCount = 0;
		int splitVar;
		double splitPoint = tree_matrix_nt[Node][7];

		if (tree_matrix_nt[Node][6] == 1) // one variable split
		{
			splitVar = (int) tree_matrix_nt[Node][8] - 1;

			if (ncat[splitVar] > 1)	// categorical variable
			{
				int* goright = (int *) malloc(ncat[splitVar]*sizeof(int));

				unpack(splitPoint, ncat[splitVar], goright);

				if (splitVar == j)
				{
					for (i =0; i<oobN; i++)
					{
						if (goright[(int) dataX_matrix[splitVar][oobObs[permuteInt[Yind[i]]]] -1] == 1)
						{
							rightNode[rightCount] = Yind[i];
							rightCount++;
						}else{
							leftNode[leftCount] = Yind[i];
							leftCount++;
						}
					}
				}else{
					for (i =0; i<oobN; i++)
					{
						if (goright[(int) dataX_matrix[splitVar][oobObs[Yind[i]]] -1] == 1)
						{
							rightNode[rightCount] = Yind[i];
							rightCount++;
						}else{
							leftNode[leftCount] = Yind[i];
							leftCount++;
						}
					}
				}
				free(goright);
			}else{
				if (splitVar == j)
				{
					for (i=0; i<oobN; i++)
					{
						if (dataX_matrix[splitVar][oobObs[permuteInt[Yind[i]]]] > splitPoint)
						{
							rightNode[rightCount] = Yind[i];
							rightCount++;
						}else{
							leftNode[leftCount] = Yind[i];
							leftCount++;
						}
					}
				}else{
					for (i=0; i<oobN; i++)
					{
						if (dataX_matrix[splitVar][oobObs[Yind[i]]] > splitPoint)
						{
							rightNode[rightCount] = Yind[i];
							rightCount++;
						}else{
							leftNode[leftCount] = Yind[i];
							leftCount++;
						}
					}
				}
			}
		}else{
			int ncomb = tree_matrix_nt[Node][6];
			int k;
			double xcomb;
			int* xvar = (int *) malloc(ncomb * sizeof(int));
			double* loading = (double *) malloc(ncomb * sizeof(double));
			int jincomb = 0;

			for (k = 0; k<ncomb; k++)
			{
				xvar[k] = (int) tree_matrix_nt[Node][8+k] - 1;
				loading[k] = tree_matrix_nt[Node][8+combsplit+k];

				if (xvar[k] == j)
					jincomb = 1;
			}

			if (jincomb == 1)
			{
				for (i=0; i<oobN; i++)
				{
					xcomb = 0;

					for (k=0; k< ncomb; k++)
					{
						if (xvar[k] == j)
						{
							xcomb += dataX_matrix[xvar[k]][oobObs[permuteInt[Yind[i]]]] * loading[k];
						}else{
							xcomb += dataX_matrix[xvar[k]][oobObs[Yind[i]]] * loading[k];
						}
					}

					if (xcomb > splitPoint)
					{
						rightNode[rightCount] = Yind[i];
						rightCount++;
					}else{
						leftNode[leftCount] = Yind[i];
						leftCount++;
					}
				}
			}else{
				for (i=0; i<oobN; i++)
				{
					xcomb = 0;

					for (k=0; k< ncomb; k++)
					{
						xcomb += dataX_matrix[xvar[k]][oobObs[Yind[i]]] * loading[k];
					}

					if (xcomb > splitPoint)
					{
						rightNode[rightCount] = Yind[i];
						rightCount++;
					}else{
						leftNode[leftCount] = Yind[i];
						leftCount++;
					}
				}
			}

			free(xvar);
			free(loading);
		}

		predict_cla_pj((int)tree_matrix_nt[Node][4], leftNode, dataX_matrix, tree_matrix_nt, combsplit, ncat, oobObs, Ypred, leftCount, permuteInt, j);
		free(leftNode);

		predict_cla_pj((int)tree_matrix_nt[Node][5], rightNode, dataX_matrix, tree_matrix_nt, combsplit, ncat, oobObs, Ypred, rightCount, permuteInt, j);
		free(rightNode);
	}
}

// score functions

double score_at_cut_cla(double* x, int* y, int* useObs, int n, int nclass, double a_random_cut)
{
	int* YLeft = (int *) calloc(nclass, sizeof(int));
	int* YRight = (int *) calloc(nclass, sizeof(int));

	int i;
	int leftcount = 0;

	for (i =0; i<n; i++)
	{
		if (x[useObs[i]] <= a_random_cut)
		{
			YLeft[y[useObs[i]]] ++;
		}else{
			YRight[y[useObs[i]]] ++;
		}
	}

	for (i=0; i< nclass; i++)
	{
		leftcount += YLeft[i];
	}

	if (leftcount > 0 && leftcount < n)
	{
		int tempLeft = 0;
		int tempRight = 0;
		for (i=0; i< nclass; i++)
		{
			tempLeft += YLeft[i]*YLeft[i];
			tempRight += YRight[i]*YRight[i];
		}

		free(YLeft);
		free(YRight);
		return((double) tempLeft/leftcount + (double) tempRight/(n - leftcount));
	}

	free(YLeft);
	free(YRight);
	return -1;
}

double score_at_cut_cla_w(double* x, int* y, double* weights, int* useObs, int n, int nclass, double a_random_cut)
{
	double* YLeft = (double *) calloc(nclass, sizeof(double));
	double* YRight = (double *) calloc(nclass, sizeof(double));

	int i;
	double leftweight = 0;
	double rightweight = 0;

	for (i =0; i<n; i++)
	{
		if (x[useObs[i]] <= a_random_cut)
		{
			YLeft[y[useObs[i]]] += weights[useObs[i]];
		}else{
			YRight[y[useObs[i]]] += weights[useObs[i]];
		}
	}

	for (i =0; i<nclass; i++)
	{
		leftweight += YLeft[i];
		rightweight += YRight[i];
	}

	if (leftweight > 0 && rightweight >0)
	{
		double tempLeft = 0;
		double tempRight = 0;
		for (i=0; i< nclass; i++)
		{
			tempLeft += YLeft[i]*YLeft[i];
			tempRight += YRight[i]*YRight[i];
		}

		free(YLeft);
		free(YRight);
		return(tempLeft/leftweight + tempRight/rightweight);
	}

	free(YLeft);
	free(YRight);
	return -1;
}

double score_at_rank_cla(struct_xc* xy, int n, int nclass, int rank)
{
	int* YLeft = (int *) calloc(nclass, sizeof(int));
	int* YRight = (int *) calloc(nclass, sizeof(int));

	int i;
	int leftcount = 0;

	for (i=0; i<rank; i++)
	{
		leftcount++;
		YLeft[xy[i].y]++;
	}

	for (i=rank; i<n; i++)
	{
		YRight[xy[i].y]++;
	}

	if (leftcount > 0 && leftcount < n)
	{
		int tempLeft = 0;
		int tempRight = 0;
		for (i=0; i< nclass; i++)
		{
			tempLeft += YLeft[i]*YLeft[i];
			tempRight += YRight[i]*YRight[i];
		}
		free(YLeft);
		free(YRight);
		return((double) tempLeft/leftcount + (double) tempRight/(n - leftcount));
	}

	free(YLeft);
	free(YRight);
	return -1;
}

double score_at_rank_cla_w(struct_xcw* xyw, int n, int nclass, int rank)
{
	double* YLeft = (double *) calloc(nclass, sizeof(double));
	double* YRight = (double *) calloc(nclass, sizeof(double));

	int i;
	double leftweight = 0;
	double rightweight = 0;

	for (i=0; i<rank; i++)
	{
		leftweight += xyw[i].w;
		YLeft[xyw[i].y] += xyw[i].w;
	}

	for (i=rank; i<n; i++)
	{
		rightweight += xyw[i].w;
		YRight[xyw[i].y] += xyw[i].w;
	}

	if (leftweight > 0 && rightweight > 0)
	{
		double tempLeft = 0;
		double tempRight = 0;
		for (i=0; i< nclass; i++)
		{
			tempLeft += YLeft[i]*YLeft[i];
			tempRight += YRight[i]*YRight[i];
		}

		free(YLeft);
		free(YRight);
		return(tempLeft/leftweight + tempRight/rightweight);
	}
	free(YLeft);
	free(YRight);
	return -1;
}

/*
void temp_score(double* xcopy, int* y, int* index, int n, int nclass, double* cutValue, double* score)
{
	double temp = -1;
	int i;
	int k;
	int leftcount = 0;
	int* YLeft = (int *) calloc(nclass, sizeof(int));
	int* YRight = (int *) calloc(nclass, sizeof(int));
	double tempLeft = 0;
	double tempRight = 0;

	// initialize for the first cut point
	for (i=0; i< (n-1); i++)
	{
		YLeft[y[index[i]]]++;

		if (xcopy[i] < xcopy[i+1]) // a possible cut point
			break;
	}

	leftcount = i+1;

	for (i = leftcount; i< n; i++)
	{
		YRight[y[index[i]]]++;
	}

	for (k=0; k< nclass; k++)
	{
		tempLeft += YLeft[k]*YLeft[k];
		tempRight += YRight[k]*YRight[k];
	}

	temp = (double) tempLeft/leftcount + (double) tempRight/(n - leftcount);

	if (temp > *score)
	{
		*cutValue = (xcopy[leftcount-1] + xcopy[leftcount])/2;
		*score = temp;
	}

	// go through the rest cut points

	for (i = leftcount; i< (n-1); i++)
	{
		YLeft[y[index[i]]] ++;
		YRight[y[index[i]]] --;

		if (xcopy[i] < xcopy[i+1])
		{
			leftcount = i+1;
			tempLeft = 0;
			tempRight = 0;

			for (k=0; k< nclass; k++)
			{
				tempLeft += YLeft[k]*YLeft[k];
				tempRight += YRight[k]*YRight[k];
			}

			temp = (double) tempLeft/leftcount + (double) tempRight/(n - leftcount);
			if (temp > *score)
			{
				*cutValue = (xcopy[leftcount-1] + xcopy[leftcount])/2;
				*score = temp;
			}
		}
	}

	free(YLeft);
	free(YRight);
}
 */
void score_best_cla(struct_xc* xy, int n, int nclass, double* cutValue, double* score)
{
	double temp = -1;
	int i;
	int k;
	int leftcount = 0;
	int* YLeft = (int *) calloc(nclass, sizeof(int));
	int* YRight = (int *) calloc(nclass, sizeof(int));
	double tempLeft = 0;
	double tempRight = 0;

	// initialize for the first cut point
	for (i=0; i< (n-1); i++)
	{
		YLeft[xy[i].y] ++;

		if (xy[i].x < xy[i+1].x) // a possible cut point
			break;
	}

	leftcount = i+1;

	for (i = leftcount; i< n; i++)
	{
		YRight[xy[i].y] ++;
	}

	for (k=0; k< nclass; k++)
	{
		tempLeft += YLeft[k]*YLeft[k];
		tempRight += YRight[k]*YRight[k];
	}

	temp = (double) tempLeft/leftcount + (double) tempRight/(n - leftcount);

	if (temp > *score)
	{
		*cutValue = (xy[leftcount-1].x + xy[leftcount].x)/2;
		*score = temp;
	}

	// go through the rest cut points

	for (i = leftcount; i< (n-1); i++)
	{
		YLeft[xy[i].y] ++;
		YRight[xy[i].y] --;

		if (xy[i].x < xy[i+1].x)
		{
			leftcount = i+1;
			tempLeft = 0;
			tempRight = 0;

			for (k=0; k< nclass; k++)
			{
				tempLeft += YLeft[k]*YLeft[k];
				tempRight += YRight[k]*YRight[k];
			}

			temp = (double) tempLeft/leftcount + (double) tempRight/(n - leftcount);
			if (temp > *score)
			{
				*cutValue = (xy[i].x + xy[i+1].x)/2;
				*score = temp;
			}
		}
	}

	free(YLeft);
	free(YRight);
}

void score_best_cla_w(struct_xcw* xyw, int n, int nclass, double* cutValue, double* score)
{
	double temp = -1;
	int i;
	int k;
	int leftcount;
	double leftweight = 0;
	double rightweight = 0;
	double* YLeft = (double *) calloc(nclass, sizeof(double));
	double* YRight = (double *) calloc(nclass, sizeof(double));
	double tempLeft = 0;
	double tempRight = 0;

	// initialize for the first cut point
	for (i=0; i< (n-1); i++)
	{
		YLeft[xyw[i].y] += xyw[i].w;
		leftweight += xyw[i].w;

		if (xyw[i].x < xyw[i+1].x) // a possible cut point
			break;
	}

	leftcount = i+1;

	for (i = leftcount; i< n; i++)
	{
		YRight[xyw[i].y] += xyw[i].w;
		rightweight += xyw[i].w;
	}

	for (k=0; k< nclass; k++)
	{
		tempLeft += YLeft[k]*YLeft[k];
		tempRight += YRight[k]*YRight[k];
	}

	temp = tempLeft/leftweight + tempRight/rightweight;

	if (temp > *score)
	{
		*cutValue = (xyw[leftcount-1].x + xyw[leftcount].x)/2;
		*score = temp;
	}

	// go through the rest cut points

	for (i = leftcount; i< (n-1); i++)
	{
		YLeft[xyw[i].y] += xyw[i].w;
		YRight[xyw[i].y] -= xyw[i].w;

		leftweight += xyw[i].w;
		rightweight -= xyw[i].w;

		if (xyw[i].x < xyw[i+1].x)
		{
			leftcount = i+1;
			tempLeft = 0;
			tempRight = 0;

			for (k=0; k< nclass; k++)
			{
				tempLeft += YLeft[k]*YLeft[k];
				tempRight += YRight[k]*YRight[k];
			}

			temp = tempLeft/leftweight + tempRight/rightweight;
			if (temp > *score)
			{
				*cutValue = (xyw[i].x + xyw[i+1].x)/2;
				*score = temp;
			}
		}
	}

	free(YLeft);
	free(YRight);
}



