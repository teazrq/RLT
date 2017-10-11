#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>

extern SEXP RLT_regression(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP RLT_classification(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP RLT_survival(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP RLT_regression_predict(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP RLT_classification_predict(SEXP, SEXP, SEXP, SEXP, SEXP);
extern SEXP RLT_survival_predict(SEXP, SEXP, SEXP, SEXP, SEXP, SEXP);

static const R_CallMethodDef callMethods[] = {
  {"RLT_regression",      (DL_FUNC) &RLT_regression,      7},
  {"RLT_classification",  (DL_FUNC) &RLT_classification,  7},
  {"RLT_survival",        (DL_FUNC) &RLT_survival,        9},
  {"RLT_regression_predict",      (DL_FUNC) &RLT_regression_predict,      5},
  {"RLT_classification_predict",  (DL_FUNC) &RLT_classification_predict,  5},
  {"RLT_survival_predict",        (DL_FUNC) &RLT_survival_predict,        6},
  {NULL, NULL, 0}
};

void R_init_RLT(DllInfo *info)
{
  R_registerRoutines(info, NULL, callMethods, NULL, NULL);
  R_useDynamicSymbols(info, TRUE);
}
