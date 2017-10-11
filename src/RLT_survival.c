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

void Fit_Trees_survival(double** dataX_matrix,
						  int* dataY_vector,
						  int* dataCensor_vector,
						  double* dataInterval_vector,
						  double*** tree_matrix,
						  double*** surv_matrix,
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
						  int use_p,
						  int nfail_unique)
{

	// get all needed parameters:

	int ntrees = myPara->ntrees;
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

	useCores = imin(ntrees, useCores);

	int haveCores = omp_get_num_procs();

	if(useCores <= 0)
	{
		useCores = 1;
		if (summary >= 2)
			Rprintf("Use at least 1 core. \n");
	}

	if(useCores > haveCores)
	{
		if (summary >= 2)
			Rprintf("Do not have %i cores, use maximum %i cores. \n", useCores, haveCores);

		useCores = haveCores;
	}

	omp_set_num_threads(useCores);

	// parallel computing:

	#pragma omp parallel private(nt, i, j, k)
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

			tree_matrix[nt][0] = (double *) malloc( (8 + 2*combsplit) * sizeof(double));
			if (tree_matrix[nt][0] == NULL) error("Unable to malloc for tree matrix");

			Split_A_Node_survival(0, inbagObs, dataX_matrix, dataY_vector, dataCensor_vector, dataInterval_vector, tree_matrix[nt], surv_matrix[nt], myPara, ncat, subjectweight, variableweight, usevariable, protectvariable, resample_size, use_p, nfail_unique);

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

				double** HazardPred = (double **) malloc (oobag_n * sizeof(double*)); // this will be just pointers to the suvival matrix, do not change its value

				int* Yind = (int *) malloc (oobag_n * sizeof(int));

				for (i=0; i<oobag_n; i++)
					Yind[i] = i;

				// non-premuted error
				predict_surv(0, Yind, dataX_matrix, tree_matrix[nt], surv_matrix[nt], combsplit, ncat, oobagObs, HazardPred, oobag_n);

				double Martingale_t = 0;
				double Martingale_Int = 0;

				double** Mt = (double **) malloc (oobag_n * sizeof(double*));
				for (i=0; i < oobag_n; i++)
					Mt[i] = (double *) calloc(nfail_unique+1, sizeof(double));

				// subject weighted not implimented !!!!!

				for (i=0; i<oobag_n; i++)
				{
					Martingale_t = 0;
					Mt[i][0] = 0;

					for (j = 1; j < dataY_vector[oobagObs[i]]; j++)
					{
						Martingale_t -= HazardPred[i][j];
						Martingale_Int += Martingale_t*Martingale_t;
						Mt[i][j] = Martingale_t;
					}

					if (dataCensor_vector[oobagObs[i]] == 0)
						Martingale_t -= HazardPred[i][j];
					else
						Martingale_t += 1 - HazardPred[i][j];

					Martingale_Int += (nfail_unique-j)*Martingale_t*Martingale_t;

					for (j = dataY_vector[oobagObs[i]]; j <= nfail_unique; j++)
						Mt[i][j] = Martingale_t;
				}

				AllError[nt][use_p] = Martingale_Int/oobag_n;

				double** HazardPred_perm = (double **) malloc (oobag_n * sizeof(double*)); // this will be just pointers to the suvival matrix, do not change its value

				// start to permute variable j

				int* permuteInt = (int *) malloc (oobag_n * sizeof(int));
				for (i=0; i<oobag_n; i++)
					permuteInt[i] = i;

				int pj;
				double Hazard_diff;
				double Adjust_t;
				double Adjust_Int;

				for (k=1; k<= npermute; k++)
				{
					permute(permuteInt, oobag_n);

					for (pj = 0; pj < use_p; pj++)
					{
						predict_surv_pj(0, Yind, dataX_matrix, tree_matrix[nt], surv_matrix[nt], combsplit, ncat, oobagObs, HazardPred_perm, oobag_n, permuteInt, usevariable[pj]);

						Martingale_Int = 0;
						Adjust_Int = 0;

						for (i=0; i<oobag_n; i++)
						{
							Martingale_t = 0;
							Adjust_t = 0;

							// calculate martingale and surv diff after permuation
							// we use square difference

							for (j = 1; j < dataY_vector[oobagObs[i]]; j++)
							{
								Hazard_diff = HazardPred[i][j] - HazardPred_perm[i][j];
								Adjust_t += Mt[i][j-1]*Hazard_diff;
								Adjust_Int += Adjust_t;

								Martingale_t -= HazardPred_perm[i][j];
								Martingale_Int += Martingale_t*Martingale_t;
							}

							if (dataCensor_vector[oobagObs[i]] == 0)
								Martingale_t -= HazardPred_perm[i][j];
							else
								Martingale_t += 1 - HazardPred_perm[i][j];

							Martingale_Int += (nfail_unique-j)*Martingale_t*Martingale_t;

							Hazard_diff = HazardPred[i][j] - HazardPred_perm[i][j];
							Adjust_t += Mt[i][j-1]*Hazard_diff;
							Adjust_Int += Adjust_t;

/* 							for (j = dataY_vector[oobagObs[i]]; j <= nfail_unique; j++)
							{
								Hazard_diff = HazardPred[i][j] - HazardPred_perm[i][j];
								Adjust_t += Mt[i][j-1]*Hazard_diff;
								Adjust_Int += Adjust_t;
							} */
						}

						AllError[nt][pj] += (Martingale_Int - 2*Adjust_Int)/oobag_n/AllError[nt][use_p] - 1;	//Diff* i didnt not devide this by oobag_n
					}
				}

				free(permuteInt);
				free(HazardPred);
				free(HazardPred_perm);

				for (i=0; i<oobag_n; i++)
					free(Mt[i]);
				free(Mt);

				free(Yind);

				for (j=0; j<use_p; j++)
				{
					if (use_sub_weight)
					{
						error("not implemented yet");
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

		// #pragma omp barrier

/* 		if (importance)
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
					VarImp[j] = VarImp[j]/ntrees;
				}else{
					VarImp[j] = NAN;
				}
			}
		} */
	}


	if (importance)
	{
		for (j=0; j<use_p; j++) // calculate variable importance
		{
			k = 0;

			for (nt = 0; nt < ntrees; nt++)
				if (AllError[nt][use_p] > 0)
				{
					VarImp[j] += AllError[nt][j];
					k++;
				}

			if (k > 0)
			{
				VarImp[j] = VarImp[j]/ntrees;
			}else{
				VarImp[j] = NAN;
			}
		}
	}

}

void Split_A_Node_survival(int Node,
							 int* useObs,
							 double** dataX_matrix,
							 int* dataY_vector,
							 int* dataCensor_vector,
							 double* dataInterval_vector,
							 double** tree_matrix_nt,
							 double** surv_matrix_nt,
							 PARAMETERS* myPara,
							 int* ncat,
							 double* subjectweight,
							 double* variableweight,
							 int* usevariable,
							 int* protectvariable,
							 int node_n,
							 int node_p,
							 int nfail_unique)
{
/*  FILE * Output;
	Output = fopen("error.txt", "a");
	fprintf(Output, "--- start node %i --- \n", Node);
	fclose(Output);  */

	int nmin = myPara->nmin;
	int combsplit = myPara->combsplit;
	int use_sub_weight = myPara->use_sub_weight;
	int i, j;
	int root = ((Node == 0) ? 1 : 0);

	// calculate node information

	tree_matrix_nt[Node][1] = Node;		// Node index
	tree_matrix_nt[Node][2] = node_n;	// Node size

	int node_fail = 0;
	for (i = 1; i<node_n; i++)
		node_fail += dataCensor_vector[useObs[i]];

	// Rprintf("At node %i, %i obs, %i failures \n", Node, node_n, node_fail);

/* 	if (myPara->reinforcement)
	{
		Rprintf("At node %i, %i obs, %i failures \n", Node, node_n, node_fail);
		Rprintf("Use variables: ");
		for (j=0; j< node_p; j++)
			Rprintf(" %i", usevariable[j]);
		Rprintf("  \n");

		Rprintf("protected variables: ");
		for (j=0; j< myPara->dataX_p; j++)
			if (protectvariable[j])
				Rprintf(" %i", j);
		Rprintf("  \n");
	} */



	if (node_fail < nmin)						// Terminate this node
	{

TerminateThisNode:	;

		int survNode = 0;

		while (surv_matrix_nt[survNode] != NULL)
			survNode++;

		// Rprintf("at tree node %i, use survival node %i \n", Node, survNode);

		surv_matrix_nt[survNode] = (double *) malloc((nfail_unique+1) * sizeof(double));

		surv_matrix_nt[survNode][0] = Node;

		double *Count_Fail = (double *) calloc((nfail_unique+1), sizeof(double));
		double *Count_Censor = (double *) calloc((nfail_unique+1), sizeof(double));

		// double Surv = 1;
		double inRisk;

		// estimate and save the survival function

		if (use_sub_weight)						// Node sample mean
		{
			// tree_matrix_nt[Node][3] stores the node index in the survival matrix

			inRisk = 0;

			for (i = 0; i < node_n; i++)
			{
				inRisk += subjectweight[useObs[i]];

				if (dataCensor_vector[useObs[i]] == 1)
					Count_Fail[dataY_vector[useObs[i]]] += subjectweight[useObs[i]];
				else
					Count_Censor[dataY_vector[useObs[i]]] += subjectweight[useObs[i]];
			}

			inRisk -= Count_Censor[0];

			for (j = 1; j < (nfail_unique + 1); j++)
			{
				if (inRisk > 0)
				{
					// Surv *= 1 - (double) Count_Fail[j]/inRisk;
					// surv_matrix_nt[survNode][j] = Surv;

					surv_matrix_nt[survNode][j] = Count_Fail[j]/inRisk;
					inRisk -= Count_Fail[j] + Count_Censor[j];
				}else{
					// surv_matrix_nt[survNode][j] = Surv;
					surv_matrix_nt[survNode][j] = 0;
				}
			}

			//print_d_mat_t(surv_matrix_nt, node_n, nfail_unique+1);

		}else{

			for (i = 0; i < node_n; i++)
			{
				if (dataCensor_vector[useObs[i]] == 1)
					Count_Fail[dataY_vector[useObs[i]]]++;
				else
					Count_Censor[dataY_vector[useObs[i]]]++;
			}

			inRisk = (double) node_n - Count_Censor[0];

			for (j = 1; j < (nfail_unique + 1); j++)
			{
				//Rprintf("Time %i, %f inRisk, %f fail \n", j, inRisk, Count_Fail[j]);

				if (inRisk > 0)
				{
					// Surv *= 1 - (double) Count_Fail[j]/inRisk;
					// surv_matrix_nt[survNode][j] = Surv;

					surv_matrix_nt[survNode][j] = Count_Fail[j]/inRisk;
					inRisk -= Count_Fail[j] + Count_Censor[j];
				}else{
					// surv_matrix_nt[survNode][j] = Surv;
					surv_matrix_nt[survNode][j] = 0;
				}
			}

			// print_d_mat_t(surv_matrix_nt, node_n, nfail_unique+1);

			// tree_matrix_nt[Node][3] stores the node index in the survival matrix
		}

		tree_matrix_nt[Node][0] = 2;			// Node type: Terminal

		tree_matrix_nt[Node][3] = survNode;		// Node type: Terminal

		tree_matrix_nt[Node][4] = NAN;			// Next Left
		tree_matrix_nt[Node][5] = NAN;			// Next Right
		tree_matrix_nt[Node][6] = NAN;			// Number of combination
		tree_matrix_nt[Node][7] = NAN;			// Splitting value

		for (i = 0; i < combsplit; i ++)		// Splitting variable(s) and loading(s)
		{
			tree_matrix_nt[Node][8+i] = NAN;
			tree_matrix_nt[Node][8+combsplit+i] = NAN;
		}

		free(Count_Fail);
		free(Count_Censor);

	}else									// Split this node
	{

		SplitRule* OneSplit = Find_A_Split_Survival(useObs, dataX_matrix, dataY_vector, dataCensor_vector, dataInterval_vector, myPara, ncat, subjectweight, variableweight, usevariable, protectvariable, node_n, node_p, nfail_unique, root);

		if (OneSplit->NCombinations == 0) // did not find linear combination, go back to terminate the node
		{
			free(OneSplit);
			goto TerminateThisNode;
		}

		// create observation indicators for left and right node

		tree_matrix_nt[Node][0] = 1;			// Node type: Internal

		int nextleft = Node;
		while (tree_matrix_nt[nextleft] != NULL)
			nextleft++;

		int nextright = nextleft + 1;
		while (tree_matrix_nt[nextright] != NULL)
			nextright++;

		tree_matrix_nt[Node][3] = NAN;
		tree_matrix_nt[Node][4] = nextleft;					// Next Left
		tree_matrix_nt[Node][5] = nextright;				// Next Right

		int i;
		int j;
		double SplitValue = OneSplit->SplitValue;
		int LeftSize = 0;
		int RightSize = 0;
		int LeftSize_Fail = 0;
		int RightSize_Fail = 0;
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
						RightSize_Fail += dataCensor_vector[useObs[i]];
					}else{
						useObsLeft[LeftSize] = useObs[i];
						LeftSize ++;
						LeftSize_Fail += dataCensor_vector[useObs[i]];
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
						LeftSize_Fail += dataCensor_vector[useObs[i]];
					}else{
						useObsRight[RightSize] = useObs[i];
						RightSize ++;
						RightSize_Fail += dataCensor_vector[useObs[i]];
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
					LeftSize_Fail += dataCensor_vector[useObs[i]];
				}else{
					useObsRight[RightSize] = tempobs;
					RightSize ++;
					RightSize_Fail += dataCensor_vector[useObs[i]];
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

		// Rprintf("Found splitting variable %i, with value %f, left node %i obs %i fail, right node %i obs, %i fail \n", OneSplit->OneSplitVariable, OneSplit->SplitValue, LeftSize, LeftSize_Fail, RightSize, RightSize_Fail);

		// just be careful here...
		if (LeftSize_Fail == 0 || RightSize_Fail == 0)
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

		tree_matrix_nt[nextleft] = (double *) malloc( (8 + 2*combsplit) * sizeof(double));
		if (tree_matrix_nt[nextleft] == NULL) error("Unable to malloc for tree matrix");

		tree_matrix_nt[nextright] = (double *) malloc( (8 + 2*combsplit) * sizeof(double));
		if (tree_matrix_nt[nextright] == NULL) error("Unable to malloc for tree matrix");

		// fit left daughter node
		Split_A_Node_survival(nextleft, useObsLeft, dataX_matrix, dataY_vector, dataCensor_vector, dataInterval_vector, tree_matrix_nt, surv_matrix_nt, myPara, ncat, subjectweight, variableweight,
								OneSplit->MutingUpdate ? OneSplit->NewUseVariable : usevariable,
								OneSplit->ProtectUpdate ? OneSplit->NewProtectVariable : protectvariable, LeftSize,
								OneSplit->MutingUpdate ? OneSplit->newp : node_p, nfail_unique);
		free(useObsLeft);

		// fit right daughter node
		Split_A_Node_survival(nextright, useObsRight, dataX_matrix, dataY_vector, dataCensor_vector, dataInterval_vector, tree_matrix_nt, surv_matrix_nt, myPara, ncat, subjectweight, variableweight,
								OneSplit->MutingUpdate ? OneSplit->NewUseVariable : usevariable,
								OneSplit->ProtectUpdate ? OneSplit->NewProtectVariable : protectvariable, RightSize,
								OneSplit->MutingUpdate ? OneSplit->newp : node_p, nfail_unique);
		free(useObsRight);

		if (OneSplit->MutingUpdate)
			free(OneSplit->NewUseVariable);

		if (OneSplit->ProtectUpdate)
			free(OneSplit->NewProtectVariable);

		free(OneSplit);

	}
}




SplitRule* Find_A_Split_Survival(int* useObs,
								   double** dataX_matrix,
								   int* dataY_vector,
								   int* dataCensor_vector,
								   double* dataInterval_vector,
								   PARAMETERS* myPara,
								   int* ncat,
								   double* subjectweight,
								   double* variableweight,
								   int* usevariable,
								   int* protectvariable,
								   int node_n,
								   int node_p,
								   int nfail_unique,
								   int root)
{
	// Get parameters
	int nmin = myPara->nmin;
	int mtry = myPara->mtry;
	int split_gen = myPara->split_gen;
	int nspliteach = myPara->nspliteach;
	int select_method = myPara->select_method;
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

	int node_fail = 0;
	for (i = 1; i<node_n; i++)
		node_fail += dataCensor_vector[useObs[i]];

	if (reinforcement & (node_fail >= 2*nmin) & (node_p > 1)) // use reinforcement learning
	{
/* 		Rprintf("fit with reinforcement learning with sample size %i and variables %i. List of variables: \n", node_n, node_p);
		for (j=0; j< node_p; j++) Rprintf(" %i  ", usevariable[j]);
		Rprintf("\n"); */

		// Rprintf("Start reinforcement...");

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

		// create matrices for fitted survivals
		double ***surv_matrix = (double ***) malloc(ntrees * sizeof(double **));
		if (surv_matrix == NULL) error("Unable to malloc for tree matrix");

		for (nt=0; nt<ntrees; nt++)
		{
			surv_matrix[nt] = (double **) malloc(node_fail * sizeof(double*));
			if (surv_matrix[nt] == NULL) error("Unable to malloc for tree matrix");

			for (i = 0; i < node_fail; i++)
				surv_matrix[nt][i] = NULL; // if this is NULL, then is node is not used yet
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

		Fit_Trees_survival(dataX_matrix, dataY_vector, dataCensor_vector, dataInterval_vector, tree_matrix, surv_matrix, AllError, VarImp, obs_in_trees, myPara_embed, ncat, subjectweight, variableweight, useObs, usevariable, protectvariable, node_n, node_p, nfail_unique);

		// remove unused objects
		for (nt = 0; nt<ntrees; nt++)
		{
			for (i = 0; i < TreeMaxLength; i++)
				if (tree_matrix[nt][i] != NULL)
					free(tree_matrix[nt][i]);

			free(tree_matrix[nt]);

			for (i = 0; i < node_fail; i++)
				if (surv_matrix[nt][i] != NULL)
					free(surv_matrix[nt][i]);

			free(surv_matrix[nt]);

			free(AllError[nt]);
		}
		free(tree_matrix);
		free(surv_matrix);
		free(AllError);
		free(myPara_embed);

		int* index = (int *) malloc(node_p * sizeof(int));
		for (j = 0; j<node_p; j++)
			index[j] = usevariable[j];

		// print_i_d_vec(index, VarImp, node_p);

		rsort_with_index(VarImp, index, node_p);

		//Rprintf("get embeded importance: \n");
		//print_i_d_vec(index, VarImp, node_p);

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

		// otherwise, I look at the variable(s) with the best VI

		if (myPara->random_select > 0)
		{
			int random_select = myPara->random_select;

			for (j =0; j < random_select; j++)
			{
				if (VarImp[node_p-1-j] > 0)
					OneSplit->NewProtectVariable[index[node_p-1-j]] = 1;
				else
					random_select = j;
			}
	/* 		for (j =0; j < random_select; j++)
				Rprintf("Variable %i, VI %f \n", index[node_p-1-j], VarImp[node_p-1-j]); */

			standardize(&VarImp[node_p-random_select], random_select); // we only standerdize the last random_select entries, the rest of the values will never be used again.

			// Rprintf("A variable is using %i \n", index[sample(&VarImp[node_p-random_select], random_select) + node_p - random_select]);

			use_var = index[sample(&VarImp[node_p-random_select], random_select) + node_p - random_select];

/* 			Rprintf("the top varialbes are:");
			for (j =0; j < random_select; j++)
				Rprintf("Variable %i, VI %f \n", index[node_p-1-j], VarImp[node_p-1-j]);
			Rprintf("\n");

			Rprintf("Use variable %i to split \n \n \n", use_var); */


		}else{
			use_var = index[node_p-1];
		}

		if (ncat[use_var] > 1)
		{
			OneSplit_Cat_Survival(&bestValue, &bestScore, dataX_matrix[use_var], dataY_vector, dataCensor_vector, subjectweight, useObs, use_sub_weight, ncat[use_var], node_n, nfail_unique, split_gen == 4 ? 4 : 3, nspliteach, select_method, nmin);
		}else{
			OneSplit_Cont_Survival(&bestValue, &bestScore, dataX_matrix[use_var], dataY_vector, dataCensor_vector, subjectweight, useObs, use_sub_weight, node_n, nfail_unique, split_gen == 4 ? 4 : 3, nspliteach, select_method, nmin);
		}

		OneSplit->NCombinations = 1;
		OneSplit->OneSplitVariable = use_var;
		OneSplit->SplitValue = bestValue;
		OneSplit->NewProtectVariable[use_var] = 1;

		// use index to indicate muted variables, then copy unmuted to the splitting rule

		use_var = 0; // use this to denote number of muted
		bestVar = 0; // use this to denote number of used

		if (myPara->muting == -1)
			use_var = imin(node_p-1, node_p*myPara->muting_percent);

		if (myPara->muting >= 1)
			use_var = imin(node_p-1, myPara->muting);

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

		// Rprintf("End: At this node, sample size %i, failure %i, found splitting var %i, value %f, score %f \n", node_n, node_fail, OneSplit->OneSplitVariable, OneSplit->SplitValue, bestScore);

	}else{ // a regular split
		// start to find splits

DoASingleSplit:; // if embedded model did not find anything, come here

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

		while(real_mtry > 0)	// randomly sample a variable
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
					OneSplit_Cat_Survival(&tempValue, &tempScore, dataX_matrix[use_var], dataY_vector, dataCensor_vector, subjectweight, useObs, use_sub_weight, x_cat, node_n, nfail_unique, split_gen, nspliteach, select_method, nmin);
					// Rprintf("cut at %f, get score %f \n", tempValue, tempScore);
				}else{
					// Rprintf("Variable %i is a continuous variable \n", use_var+1);
					OneSplit_Cont_Survival(&tempValue, &tempScore, dataX_matrix[use_var], dataY_vector, dataCensor_vector, subjectweight, useObs, use_sub_weight, node_n, nfail_unique, split_gen, nspliteach, select_method, nmin);
					// Rprintf("cut at %f, get score %f \n", tempValue, tempScore);
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


void OneSplit_Cont_Survival(double *cutValue, double* score, double* x, int* y, int* c, double* weights, int* useObs, int use_weight, int n, int nfail, int split_gen, int nspliteach, int select_method, int nmin)
{
	*cutValue = NAN;
	*score = -1;

	int i, j;
	double a_random_cut;
	double a_random_score = -1;

	if (split_gen == 1) // random
	{
		if (use_weight)
		{
			for (i=0; i<nspliteach; i++)
			{
				a_random_cut = x[useObs[random_in_range(0, n)]];
				a_random_score = score_at_cut_surv_w(x, y, c, weights, useObs, n, nfail, a_random_cut, select_method);

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
				a_random_score = score_at_cut_surv(x, y, c, useObs, n, nfail, a_random_cut, select_method);

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
				a_random_score = score_at_cut_surv_w(x, y, c, weights, useObs, n, nfail, a_random_cut, select_method);

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
				a_random_score = score_at_cut_surv(x, y, c, useObs, n, nfail, a_random_cut, select_method);

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
			error("Cannot do weighted split yet");
		}else{

			// Rprintf("searching for a ranked split");

			// copy and sort
			struct_xyc* copy_xyc = (struct_xyc*) malloc(n * sizeof(struct_xyc));
			for (i =0; i< n; i++)
			{
				copy_xyc[i].x = x[useObs[i]];
				copy_xyc[i].y = y[useObs[i]];
				copy_xyc[i].c = c[useObs[i]];
			}

			qsort(copy_xyc, n, sizeof(struct_xyc), compare_struct_xyc);

			int lower_first = 0;
			int upper_first = n-1;

			i=0;
			j=0;

			while (i < 1 && j < 1)
			{
				if (i <= j)
				{
					i += copy_xyc[lower_first++].c;
				}

				if (i >= j)
				{
					j += copy_xyc[upper_first--].c;
				}
			}

			// this is a weird situation where we cannot acturally find a split...
			if (copy_xyc[lower_first].x == copy_xyc[n-1].x || copy_xyc[0].x == copy_xyc[upper_first].x)
			{
				free(copy_xyc);
				return;
			}

			int lower = lower_first;
			int upper = upper_first;

			while (i < (int) nmin/2 && j < (int) nmin/2)
			{
				if (i <= j)
				{
					i += copy_xyc[lower++].c;
				}

				if (i >= j)
				{
					j += copy_xyc[upper--].c;
				}
			}

			// Rprintf("get lower value %i and upper value %i \n", lower, upper+1);

			for (i=0; i<nspliteach; i++)
			{
				a_random_rank = random_in_range(lower, upper); // get a random rank for split

				// adjust for ties
				if (copy_xyc[a_random_rank].x == copy_xyc[0].x)
				{
					while (copy_xyc[a_random_rank].x == copy_xyc[0].x) a_random_rank++;
				}else if (copy_xyc[a_random_rank-1].x == copy_xyc[n-1].x)
				{
					while (copy_xyc[a_random_rank-1].x == copy_xyc[n-1].x) a_random_rank--;
				}else if (unif_rand() > 0.5) // while in the middle of a sequence of ties, either move up or move down
				{
					while (copy_xyc[a_random_rank].x == copy_xyc[a_random_rank-1].x) a_random_rank++;
				}else{
					while (copy_xyc[a_random_rank].x == copy_xyc[a_random_rank-1].x) a_random_rank--;
				}

				// get cut and score

				a_random_cut = (copy_xyc[a_random_rank-1].x + copy_xyc[a_random_rank].x)/2;
				a_random_score = score_at_rank_surv(copy_xyc, n, nfail, a_random_rank, select_method);

				if (a_random_score > *score)
				{
					*cutValue = a_random_cut;
					*score = a_random_score;
				}
			}

			free(copy_xyc);
		}

		return;
	}


	if (split_gen == 4) // best
	{
		double LeftN, AllN;
		double numerator, denominator;

		if (use_weight)
		{
			error("Cannot do weighted split yet");
		}else{

			// Rprintf("searching for a ranked split");

			// copy and sort
			struct_xyc* copy_xyc = (struct_xyc*) malloc(n * sizeof(struct_xyc));
			for (i =0; i< n; i++)
			{
				copy_xyc[i].x = x[useObs[i]];
				copy_xyc[i].y = y[useObs[i]];
				copy_xyc[i].c = c[useObs[i]];
			}

			qsort(copy_xyc, n, sizeof(struct_xyc), compare_struct_xyc);

			int lower_first = 0;
			int upper_first = n-1;

			i=0;
			j=0;

			while (i < 1 && j < 1)
			{
				if (i <= j)
				{
					i += copy_xyc[lower_first++].c;
				}

				if (i >= j)
				{
					j += copy_xyc[upper_first--].c;
				}
			}

			// this is a weird situation where we cannot acturally find a split...
			if (copy_xyc[lower_first].x == copy_xyc[n-1].x || copy_xyc[0].x == copy_xyc[upper_first].x)
			{
				free(copy_xyc);
				return;
			}

			int lower = lower_first;
			int upper = upper_first;

			while (i < (int) nmin/2 && j < (int) nmin/2)
			{
				if (i <= j)
				{
					i += copy_xyc[lower++].c;
				}

				if (i >= j)
				{
					j += copy_xyc[upper--].c;
				}
			}

			// Rprintf("get lower value %i and upper value %i \n", lower, upper+1);

			int* Left_Count_Censor = (int *) calloc(nfail+1, sizeof(int));
			int* Left_Count_Fail = (int *) calloc(nfail+1, sizeof(int));
			int* Right_Count_Censor = (int *) calloc(nfail+1, sizeof(int));
			int* Right_Count_Fail = (int *) calloc(nfail+1, sizeof(int));

			for (i = 0; i< lower; i++)
			{
				if (copy_xyc[i].c == 1)
					Left_Count_Fail[copy_xyc[i].y]++;
				else
					Left_Count_Censor[copy_xyc[i].y]++;
			}

			for (i= lower; i < n; i++)
			{
				if (copy_xyc[i].c == 1)
					Right_Count_Fail[copy_xyc[i].y]++;
				else
					Right_Count_Censor[copy_xyc[i].y]++;
			}

			for (i = lower; i < upper; i++)
			{
				while (copy_xyc[i-1].x == copy_xyc[i].x)
				{
					if (copy_xyc[i].c == 1)
					{
						Left_Count_Fail[copy_xyc[i].y]++;
						Right_Count_Fail[copy_xyc[i].y]--;
					}else{
						Left_Count_Censor[copy_xyc[i].y]++;
						Right_Count_Censor[copy_xyc[i].y]--;
					}
					i++;
				}

				LeftN = (double) i - Left_Count_Censor[0];
				AllN = (double) n - Left_Count_Censor[0] - Right_Count_Censor[0];

				numerator = 0;
				denominator = 0;

				if (select_method == 3)
				{
					for (j = 1; j < (nfail + 1) && AllN > 1; j++)
					{
						numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];
						denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

						// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

						LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
						AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
					}

					if (denominator > 0)
					{
						if (numerator*numerator/denominator > *score)
						{
							*cutValue = (copy_xyc[i-1].x + copy_xyc[i].x)/2;
							*score = numerator*numerator/denominator;
						}
					}
				}

				if (select_method == 4)
				{
					for (j = 1; j < (nfail + 1) && AllN > 1; j++)
					{
						numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];
						denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

						// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

						LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
						AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];

						if (denominator > 0)
						{
							if (numerator*numerator/denominator > *score)
							{
								*cutValue = (copy_xyc[i-1].x + copy_xyc[i].x)/2;
								*score = numerator*numerator/denominator; // copy randomRight into goright with the correct ordering
							}
						}
					}
				}
			}

			free(Left_Count_Censor);
			free(Left_Count_Fail);
			free(Right_Count_Censor);
			free(Right_Count_Fail);

			free(copy_xyc);
		}

		return;
	}
}

void OneSplit_Cat_Survival(double *cutValue, double *score, double* x, int* y, int* c, double* weights, int* useObs, int use_weight, int x_cat, int n, int nfail, int split_gen, int nspliteach, int select_method, int nmin)
{
	*cutValue = NAN;
	*score = -1;

	int i, j, k;
	int temp_cat;
	int true_x_cat;
	int true_x_fail_cat;

	// summarize this categorical variable
	struct_ifc* cat_count = (struct_ifc*) malloc(x_cat * sizeof(struct_ifc));

	for (i=0; i< x_cat; i++)
	{
		cat_count[i].cat = i;
		cat_count[i].f = 0;
		cat_count[i].c = 0;
		cat_count[i].flist = (int*) calloc(nfail+1, sizeof(int));
		cat_count[i].clist = (int*) calloc(nfail+1, sizeof(int));
	}

	if (use_weight)
	{
		error("not implemented");
	}else{
		for (i=0; i<n; i++)
		{
			temp_cat = (int) x[useObs[i]] -1;

			if (c[useObs[i]] == 0)
			{
				cat_count[temp_cat].clist[y[useObs[i]]]++;
				cat_count[temp_cat].c++;
			}else{
				cat_count[temp_cat].flist[y[useObs[i]]]++;
				cat_count[temp_cat].f++;
			}
		}
	}

	// put the nonzero categories to the front

	true_x_cat = x_cat;
	for (i =0; i < true_x_cat; i++)
	{
		if (cat_count[i].f + cat_count[i].c <= 0)
		{
			swap_ifc(&cat_count[i], &cat_count[true_x_cat-1]);
			true_x_cat--;
			i--;
		}
	}

	// put categories with nonzero failures to the front

	true_x_fail_cat = true_x_cat;

	for (i =0; i < true_x_fail_cat; i++)
	{
		if (cat_count[i].f <= 0)
		{
			swap_ifc(&cat_count[i], &cat_count[true_x_fail_cat-1]);
			true_x_fail_cat--;
			i--;
		}
	}

/* 	for (i = 0; i < x_cat; i++)
		Rprintf("Cat %i, failure %i, censoring %i \n", cat_count[i].cat, cat_count[i].f, cat_count[i].c);
	 */

	if (true_x_fail_cat <= 1)
		goto NothingToFind;

	// for the categoris that has nonzero failure observations, I randomly select some to go right

	int* goright = (int *) malloc(x_cat*sizeof(int));
	int* randomRight = (int *) malloc(x_cat*sizeof(int));
	int* Left_Count_Censor = (int *) malloc((nfail+1)*sizeof(int));
	int* Left_Count_Fail = (int *) malloc((nfail+1)*sizeof(int));
	int* Right_Count_Censor = (int *) malloc((nfail+1)*sizeof(int));
	int* Right_Count_Fail = (int *) malloc((nfail+1)*sizeof(int));
	double numerator = 0;
	double denominator = 0;
	double LeftN, AllN;

	if (split_gen == 1 || split_gen == 2 || split_gen == 3)
	{

		if (nspliteach > pow(2, true_x_fail_cat-1) && true_x_fail_cat <= 6)
			goto UseBestSplit;

UseRandomSplit:

		for (k =0; k<nspliteach; k++)
		{
			memset(goright, 0, x_cat*sizeof(int));
			memset(randomRight, 0, x_cat*sizeof(int));
			memset(Left_Count_Censor, 0, (nfail+1)*sizeof(int));
			memset(Left_Count_Fail, 0, (nfail+1)*sizeof(int));
			memset(Right_Count_Censor, 0, (nfail+1)*sizeof(int));
			memset(Right_Count_Fail, 0, (nfail+1)*sizeof(int));
			numerator = 0;
			denominator = 0;
			LeftN = 0;

			// get a random splitting rule
			for (i=0; i<true_x_fail_cat; i++)
				randomRight[i] = 1;

			memset(randomRight, 0, random_in_range(1, true_x_fail_cat)*sizeof(int));
			permute(randomRight, true_x_fail_cat);

			for (i=true_x_fail_cat; i<x_cat; i++)
				randomRight[i] = (int) unif_rand()>0.5;

			for (i = 0; i < true_x_cat; i++)
			{
				if (randomRight[i] == 0)
				{
					for (j = 1; j < (nfail+1); j++)
					{
						Left_Count_Censor[j] += cat_count[i].clist[j];
						Left_Count_Fail[j] += cat_count[i].flist[j];
					}
					LeftN += cat_count[i].f + cat_count[i].c;
				}else{
					for (j = 1; j < (nfail+1); j++)
					{
						Right_Count_Censor[j] += cat_count[i].clist[j];
						Right_Count_Fail[j] += cat_count[i].flist[j];
					}
				}
			}

			LeftN = LeftN - Left_Count_Censor[0];
			AllN = (double) n - Left_Count_Censor[0] - Right_Count_Censor[0];

			if (select_method == 3) // logrank
			{
				for (j = 1; j < (nfail + 1) && AllN > 1; j++)
				{
					numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];
					denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

					// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

					LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
					AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
				}

				// copy randomRight into goright with the correct ordering
				if (denominator > 0)
				{
					if (numerator*numerator/denominator > *score)
					{
						for (i = 0; i< x_cat; i ++)
							if (randomRight[i] == 1)
								goright[cat_count[i].cat] = 1;

						*cutValue = pack(x_cat, goright);
						*score = numerator*numerator/denominator;
					}
				}
			}


			if (select_method == 4) // sup logrank
			{
				for (j = 1; j < (nfail + 1) && AllN > 1; j++)
				{
					numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];
					denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

					// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

					LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
					AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];

					// copy randomRight into goright with the correct ordering
					if (denominator > 0)
					{
						if (numerator*numerator/denominator > *score)
						{
							for (i = 0; i< x_cat; i ++)
								if (randomRight[i] == 1)
									goright[cat_count[i].cat] = 1;

							*cutValue = pack(x_cat, goright);
							*score = numerator*numerator/denominator;
						}
					}
				}
			}

		}
	}

	if (split_gen == 4)
	{
		if (true_x_fail_cat > 6) // I dont want to handel too many categories
		{
			nspliteach = 32;
			goto UseRandomSplit;
		}

UseBestSplit:

		memset(randomRight, 0, x_cat*sizeof(int));

		nspliteach = pow(2, true_x_fail_cat-1); // I will only care about seperating the categories that has failured observations

		// this is a slower program than my classification code... Im so lazy...

		for (k = 0; k < nspliteach; k++)
		{
			memset(goright, 0, x_cat*sizeof(int));
			memset(Left_Count_Censor, 0, (nfail+1)*sizeof(int));
			memset(Left_Count_Fail, 0, (nfail+1)*sizeof(int));
			memset(Right_Count_Censor, 0, (nfail+1)*sizeof(int));
			memset(Right_Count_Fail, 0, (nfail+1)*sizeof(int));
			numerator = 0;
			denominator = 0;
			LeftN = 0;

			randomRight[0]++;

			for (i = 0; i < true_x_fail_cat; i++)
			{
				if (randomRight[i] == 1)
				{
					randomRight[i] = 0;
					randomRight[i+1] ++;
				}
			}

			for (i=true_x_fail_cat; i<x_cat; i++)  // the empty categories are randomly assigned
				randomRight[i] = (int) unif_rand()>0.5;

			for (i = 0; i < true_x_cat; i++)
			{
				if (randomRight[i] == 0)
				{
					for (j = 1; j < (nfail+1); j++)
					{
						Left_Count_Censor[j] += cat_count[i].clist[j];
						Left_Count_Fail[j] += cat_count[i].flist[j];
					}
					LeftN += cat_count[i].f + cat_count[i].c;
				}else{
					for (j = 1; j < (nfail+1); j++)
					{
						Right_Count_Censor[j] += cat_count[i].clist[j];
						Right_Count_Fail[j] += cat_count[i].flist[j];
					}
				}
			}

			LeftN = LeftN - Left_Count_Censor[0];
			AllN = (double) n - Left_Count_Censor[0] - Right_Count_Censor[0];

			if (select_method == 3) // logrank
			{
				for (j = 1; j < (nfail + 1) && AllN > 1; j++)
				{
					numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];
					denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

					// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

					LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
					AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
				}
					// copy randomRight into goright with the correct ordering
				if (denominator > 0)
				{
					if (numerator*numerator/denominator > *score)
					{
						for (i = 0; i< x_cat; i ++)
							if (randomRight[i] == 1)
								goright[cat_count[i].cat] = 1;

						*cutValue = pack(x_cat, goright);
						*score = numerator*numerator/denominator;
					}
				}
			}

			if (select_method == 4) // sup logrank
			{
				for (j = 1; j < (nfail + 1) && AllN > 1; j++)
				{
					numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];
					denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

					// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

					LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
					AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];

					// copy randomRight into goright with the correct ordering
					if (denominator > 0)
					{
						if (numerator*numerator/denominator > *score)
						{
							for (i = 0; i< x_cat; i ++)
								if (randomRight[i] == 1)
									goright[cat_count[i].cat] = 1;

							*cutValue = pack(x_cat, goright);
							*score = numerator*numerator/denominator;
						}
					}
				}
			}
		}
	}

	free(goright);
	free(Left_Count_Censor);
	free(Left_Count_Fail);
	free(Right_Count_Censor);
	free(Right_Count_Fail);

NothingToFind: ;

	for (i=0; i< x_cat; i++)
	{
		free(cat_count[i].flist);
		free(cat_count[i].clist);
	}

	free(cat_count);

	return;
}

double score_at_cut_surv(double* x, int* y, int* c, int* useObs, int n, int nfail, double a_random_cut, int select_method)
{
	int *Left_Count_Fail = (int *) calloc((nfail+1), sizeof(int));
	int *Left_Count_Censor = (int *) calloc((nfail+1), sizeof(int));

	int *Right_Count_Fail = (int *) calloc((nfail+1), sizeof(int));
	int *Right_Count_Censor = (int *) calloc((nfail+1), sizeof(int));

	double LeftN = 0;
	double AllN;

	int i, j;
	double supscore = -1;

	for (i = 0; i < n; i++)
	{
		if (x[useObs[i]] <= a_random_cut)
		{
			LeftN ++;

			if (c[useObs[i]] == 1)
				Left_Count_Fail[y[useObs[i]]]++;
			else
				Left_Count_Censor[y[useObs[i]]]++;
		}else{
			if (c[useObs[i]] == 1)
				Right_Count_Fail[y[useObs[i]]]++;
			else
				Right_Count_Censor[y[useObs[i]]]++;
		}
	}

	LeftN = LeftN - Left_Count_Censor[0];
	AllN = (double) n - Left_Count_Censor[0] - Right_Count_Censor[0];
	double numerator = 0;
	double denominator = 0;


	if (select_method == 3) // logrank
	{
		for (j = 1; j < (nfail + 1) && AllN > 1; j++)
		{
			numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];

			denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

			// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

			LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
			AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
		}

		if (denominator > 0)
			if (numerator*numerator/denominator > supscore)
				supscore = numerator*numerator/denominator;
	}

	if (select_method == 4) // sup logrank
	{
		for (j = 1; j < (nfail + 1) && AllN > 1; j++)
		{
			numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];

			denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

			// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

			LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
			AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];

			if (denominator > 0)
				if (numerator*numerator/denominator > supscore)
					supscore = numerator*numerator/denominator;
		}
	}


	free(Left_Count_Fail);
	free(Left_Count_Censor);
	free(Right_Count_Fail);
	free(Right_Count_Censor);

	return supscore;

/* 	if (denominator > 0)
		return numerator*numerator/denominator;

	return -1; */
}


double score_at_cut_surv_w(double* x, int* y, int* c, double* weights, int* useObs, int n, int nfail, double a_random_cut, int select_method)
{
	return -1;
}


double score_at_rank_surv(struct_xyc* xyc, int n, int nfail, int rank, int select_method)
{
	int *Left_Count_Fail = (int *) calloc((nfail+1), sizeof(int));
	int *Left_Count_Censor = (int *) calloc((nfail+1), sizeof(int));

	int *Right_Count_Fail = (int *) calloc((nfail+1), sizeof(int));
	int *Right_Count_Censor = (int *) calloc((nfail+1), sizeof(int));

	double LeftN = 0;
	double AllN;

	int i, j;

	double supscore = -1;

	for (i=0; i<rank; i++)
	{
		LeftN ++;

		if (xyc[i].c == 1)
			Left_Count_Fail[xyc[i].y]++;
		else
			Left_Count_Censor[xyc[i].y]++;
	}

	for (i=rank; i<n; i++)
	{
			if (xyc[i].c == 1)
				Right_Count_Fail[xyc[i].y]++;
			else
				Right_Count_Censor[xyc[i].y]++;
	}

	LeftN = LeftN - Left_Count_Censor[0];
	AllN = (double) n - Left_Count_Censor[0] - Right_Count_Censor[0];
	double numerator = 0;
	double denominator = 0;

	if (select_method == 3) // logrank
	{
		for (j = 1; j < (nfail + 1) && AllN > 1; j++)
		{
			numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];

			denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

			// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

			LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
			AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];
		}

		if (denominator > 0)
			if (numerator*numerator/denominator > supscore)
				supscore = numerator*numerator/denominator;
	}

	if (select_method == 4) // sup logrank
	{
		for (j = 1; j < (nfail + 1) && AllN > 1; j++)
		{
			numerator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN - Left_Count_Fail[j];

			denominator += LeftN*(Left_Count_Fail[j] + Right_Count_Fail[j])/AllN*(1- LeftN/AllN)*(AllN - Left_Count_Fail[j] - Right_Count_Fail[j])/(AllN - 1);

			// Rprintf("leftN is %f, allN is %f, Num is %f, demo is %f \n", LeftN, AllN, numerator, denominator);

			LeftN -= Left_Count_Fail[j] + Left_Count_Censor[j];
			AllN -= Left_Count_Fail[j] + Left_Count_Censor[j] + Right_Count_Fail[j] + Right_Count_Censor[j];

			if (denominator > 0)
				if (numerator*numerator/denominator > supscore)
					supscore = numerator*numerator/denominator;
		}
	}

	free(Left_Count_Fail);
	free(Left_Count_Censor);
	free(Right_Count_Fail);
	free(Right_Count_Censor);

	return supscore;

/* 	if (denominator > 0)
		return numerator*numerator/denominator;

	return -1; */
}


double score_at_rank_surv_w(struct_xycw* xycw, int n, int nfail, int rank, int select_method)
{
	return -1;
}

// prediction functions

void predict_surv(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, double** surv_matrix_nt, int combsplit, int* ncat, int* oobObs, double** SurvPred, int oobN)
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
		int survNode = tree_matrix_nt[Node][3];

		for (i=0; i<oobN; i++)
			SurvPred[Yind[i]] = surv_matrix_nt[survNode];

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
		predict_surv((int)tree_matrix_nt[Node][4], leftNode, dataX_matrix, tree_matrix_nt, surv_matrix_nt, combsplit, ncat, oobObs, SurvPred, leftCount);
		free(leftNode);

		predict_surv((int)tree_matrix_nt[Node][5], rightNode, dataX_matrix, tree_matrix_nt, surv_matrix_nt, combsplit, ncat, oobObs, SurvPred, rightCount);
		free(rightNode);
	}
}

void predict_surv_pj(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, double** surv_matrix_nt, int combsplit, int* ncat, int* oobObs, double** SurvPred, int oobN, int* permuteInt, int j)
{
	int i;

	if (tree_matrix_nt[Node][0] == 2)
	{
		int survNode = tree_matrix_nt[Node][3];

		for (i=0; i<oobN; i++)
			SurvPred[Yind[i]] = surv_matrix_nt[survNode];

	}else{

		int* leftNode = (int *) malloc(oobN * sizeof(int));
		int* rightNode = (int *) malloc(oobN * sizeof(int));
		int leftCount = 0;
		int rightCount = 0;
		int splitVar;
		double splitPoint = tree_matrix_nt[Node][7];

		if (tree_matrix_nt[Node][6]== 1) // one variable split
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

		predict_surv_pj((int)tree_matrix_nt[Node][4], leftNode, dataX_matrix, tree_matrix_nt, surv_matrix_nt, combsplit, ncat, oobObs, SurvPred, leftCount, permuteInt, j);
		free(leftNode);

		predict_surv_pj((int)tree_matrix_nt[Node][5], rightNode, dataX_matrix, tree_matrix_nt, surv_matrix_nt, combsplit, ncat, oobObs, SurvPred, rightCount, permuteInt, j);
		free(rightNode);
	}
}


