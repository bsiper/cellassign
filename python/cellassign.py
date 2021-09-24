import tensorflow as tf
import numpy as np
from .utils import *
import logging

logger = logging.getLogger(__name__)


class CellAssign:
    def __init__(self,
                 exprs_obj, 
                 marker_gene_info, 
                 s=None, 
                 X=None, 
                 min_delta=2,
                 B=10,
                 shrinkage=True,
                 n_batches=1,
                 dirichlet_concentration=1e-2,
                 rel_tol_adam=1e-4,
                 rel_tol_em=1e-4,
                 max_iter_adam = 1e5,
                 max_iter_em = 20,
                 learning_rate = 0.1,
                 verbose = True,
                 sce_assay = "counts",
                 return_SCE = False,
                 num_runs = 1,
                 threads = 0):
        self.exprs_obj = exprs_obj
        self.marker_gene_info = marker_gene_info
        self.s = s
        self.X = X
        self.min_delta = min_delta
        self.B = B
        self.shrinkage = shrinkage
        self.n_batches = n_batches
        self.dirichlet_concentration = dirichlet_concentration
        self.rel_tol_adam = rel_tol_adam
        self.rel_tol_em = rel_tol_em
        self.max_iter_adam = max_iter_adam
        self.max_iter_em = max_iter_em
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.sce_assay = sce_assay
        self.return_SCE = return_SCE
        self.num_runs = num_runs
        self.threads = threads
    
    def run(self):
        # work out rho
        if isinstance(self.marker_gene_info, (tf.Tensor, np.ndarray)):
            rho = self.marker_gene_info
        elif isinstance(self.marker_gene_info, list):
            rho = marker_list_to_mat(self.marker_gene_info, include_other=False)
        else:
            raise Exception("marker_gene_info must either be a matrix or list. See cellassign -h")
        
        # get expression input
        Y = extract_expression_matrix(self.exprs_obj, sce_assay=self.sce_assay)

        # check types
        if self.X is not None:
            if not isinstance(self.X, (tf.Tensor, np.ndarray)):
                raise Exception("X must either be None or a numeric matrix")
        
        if s is not None:
            if isinstance(s, list):
                # TODO: Assuming extract_expression_matrix returns numpy ndarray
                # If list of s is same length as Y rows
                if len(s) != np.shape(Y)[0]:
                    raise Exception("Y has different row count than s")
            else:
                raise Exception("s is not list")
        
        # This does colSums (rewritten for numpy array) over a transposed Y's rows for cols
        if any(np.sum(y) == 0 for y in Y.T):
            logger.warning("Genes with no mapping counts are present. Make sure this is expected -- this can be valid input in some cases (e.g. when cell types are overspecified).")

        if any(np.sum(y) == 0 for y in Y):
            logger.warning("Cells with no mapping counts are present. You might want to filter these out prior to using cellassign.")
        



