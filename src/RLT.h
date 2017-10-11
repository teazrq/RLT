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

#ifndef RLT_reg
#define RLT_reg

// regression model

SEXP RLT_regression(SEXP datasetX_R, SEXP datasetY_R, SEXP ncat_R, SEXP subjectweight_R, SEXP variableweight_R, SEXP parameters_int_R, SEXP parameters_double_R);

void Fit_Trees_regression(double** dataX_matrix, double* dataY_vector, double*** tree_matrix, double** AllMSE, double *VarImp, int** obs_in_trees, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* obsIndicator, int* usevariable, int* protectvariable, int use_n, int node_p);

void Split_A_Node_regression(int Node, int* useObs, double** dataX_matrix, double* dataY_vector, double** tree_matrix_nt, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* usevariable, int* protectvariable, int node_n, int node_p);

SplitRule* Find_A_Split_Regression(int* useObs, double** dataX_matrix, double* dataY_vector, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* usevariable, int* protectvariable, int node_n, int node_p, int root);

void OneSplit_Cat_Regression(double *cutpoint, double *score, double* x, double* y, double* weights, int* useObs, int use_weight, int x_cat, int n, int split_gen, int nspliteach, int nmin);

void OneSplit_Cont_Regression(double *cutpoint, double *score, double* x, double* y, double* weights, int* useObs, int use_weight, int n, int split_gen, int nspliteach, int nmin);

double score_at_rank_reg(struct_xy* xyw, int n, int rank);
double score_at_rank_reg_w(struct_xyw* xyw, int n, int rank);
double score_at_cut_reg(double* x, double* y, int* useObs, int n, double a_random_cut);
double score_at_cut_reg_w(double* x, double* y, double* weights, int* useObs, int n, double a_random_cut);
void score_best_reg(struct_xy* xy, int n, double* cutValue, double* score);
void score_best_reg_w(struct_xyw* xyw, int n, double* cutValue, double* score);


SEXP RLT_regression_predict(SEXP datasetX_R, SEXP FittedTrees_R, SEXP ncat_R, SEXP parameters_int_R, SEXP parameters_double_R);

void predict_reg(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int combsplit, int* ncat, int* oobObs, double* Ypred, int oobN);
void predict_reg_pj(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int combsplit, int* ncat, int* oobObs, double* Ypred, int oobN, int* permuteInt, int j);

// classification model

SEXP RLT_classification(SEXP datasetX_R, SEXP datasetY_R, SEXP ncat_R, SEXP subjectweight_R, SEXP variableweight_R, SEXP parameters_int_R, SEXP parameters_double_R);

void Fit_Trees_classification(double** dataX_matrix, int* dataY_vector, double*** tree_matrix, double** AllMSE, double *VarImp, int** obs_in_trees, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* obsIndicator, int* usevariable, int* protectvariable, int use_n, int node_p);

void Split_A_Node_classification(int Node, int* useObs, double** dataX_matrix, int* dataY_vector, double** tree_matrix_nt, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* usevariable, int* protectvariable, int node_n, int node_p);

SplitRule* Find_A_Split_classification(int* useObs, double** dataX_matrix, int* dataY_vector, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* usevariable, int* protectvariable, int node_n, int node_p, int root);

void OneSplit_Cat_classification(double *cutpoint, double *score, double* x, int* y, double* weights, int* useObs, int use_weight, int x_cat, int n, int nclass, int split_gen, int nspliteach, int nmin);
void OneSplit_Cont_classification(double *cutpoint, double *score, double* x, int* y, double* weights, int* useObs, int use_weight, int n, int nclass, int split_gen, int nspliteach, int nmin);
double score_at_cut_cla(double* x, int* y, int* useObs, int n, int nclass, double a_random_cut);
double score_at_cut_cla_w(double* x, int* y, double* weights, int* useObs, int n, int nclass, double a_random_cut);
double score_at_rank_cla(struct_xc* xy, int n, int nclass, int rank);
double score_at_rank_cla_w(struct_xcw* xyw, int n, int nclass, int rank);
void score_best_cla(struct_xc* xy, int n, int nclass, double* cutValue, double* score);
void score_best_cla_w(struct_xcw* xyw, int n, int nclass, double* cutValue, double* score);

// void temp_score(double* xcopy, int* y, int* index, int n, int nclass, double* cutValue, double* score);

SEXP RLT_classification_predict(SEXP datasetX_R, SEXP FittedTrees_R, SEXP ncat_R, SEXP parameters_int_R, SEXP parameters_double_R);

void predict_cla(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int combsplit, int* ncat, int* oobObs, int* Ypred, int oobN);
void predict_cla_pj(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int combsplit, int* ncat, int* oobObs, int* Ypred, int oobN, int* permuteInt, int j);

void predict_cla_all(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, int nclass, int combsplit, int* ncat, int* oobObs, double** Ypred, int oobN);


// survival model

SEXP RLT_survival(SEXP datasetX_R, SEXP datasetY_R, SEXP datasetCensor_R, SEXP datasetInterval_R, SEXP ncat_R, SEXP subjectweight_R, SEXP variableweight_R, SEXP parameters_int_R, SEXP parameters_double_R);

void Fit_Trees_survival(double** dataX_matrix, int* dataY_vector, int* dataCensor_vector, double* dataInterval_vector, double*** tree_matrix, double*** surv_matrix, double** AllError, double* VarImp, int** obs_in_trees, PARAMETERS* myPara,
						  int* ncat, double* subjectweight, double* variableweight, int* obsIndicator, int* usevariable, int* protectvariable, int use_n, int use_p, int nfail_unique);


void Split_A_Node_survival(int Node, int* useObs, double** dataX_matrix, int* dataY_vector, int* dataCensor_vector, double* dataInterval_vector, double** tree_matrix_nt, double** surv_matrix_nt, PARAMETERS* myPara,
							 int* ncat, double* subjectweight, double* variableweight, int* usevariable, int* protectvariable, int node_n, int node_p, int nfail_unique);

SplitRule* Find_A_Split_Survival(int* useObs, double** dataX_matrix, int* dataY_vector, int* dataCensor_vector, double* dataInterval_vector, PARAMETERS* myPara,
						int* ncat, double* subjectweight, double* variableweight, int* usevariable, int* protectvariable, int node_n, int node_p, int nfail_unique, int root);


void OneSplit_Cont_Survival(double *cutValue, double* score, double* x, int* y, int* c, double* weights, int* useObs, int use_weight, int n, int nfail, int split_gen, int nspliteach, int select_method, int nmin);
void OneSplit_Cat_Survival(double *cutValue, double *score, double* x, int* y, int* c, double* weights, int* useObs, int use_weight, int x_cat, int n, int nfail, int split_gen, int nspliteach, int select_method, int nmin);

double score_at_cut_surv(double* x, int* y, int* c, int* useObs, int n, int nfail, double a_random_cut, int select_method);
double score_at_cut_surv_w(double* x, int* y, int* c, double* weights, int* useObs, int n, int nfail, double a_random_cut, int select_method);

double score_at_rank_surv(struct_xyc* xyc, int n, int nfail, int rank, int select_method);
double score_at_rank_surv_w(struct_xycw* xycw, int n, int nfail, int rank, int select_method);

SEXP RLT_survival_predict(SEXP datasetX_R, SEXP FittedTrees_R, SEXP FittedSurv_R, SEXP ncat_R, SEXP parameters_int_R, SEXP parameters_double_R);

void predict_surv(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, double** surv_matrix_nt, int combsplit, int* ncat, int* oobObs, double** SurvPred, int oobN);
void predict_surv_pj(int Node, int* Yind, double** dataX_matrix, double** tree_matrix_nt, double** surv_matrix_nt, int combsplit, int* ncat, int* oobObs, double** SurvPred, int oobN, int* permuteInt, int j);
#endif









