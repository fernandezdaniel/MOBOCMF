
import numpy as np
import torch
import scipy.optimize as spo


from scipy.spatial import KDTree
from scipy.spatial.distance import cdist


class NotFeasiblePoints(ValueError):
    pass

class MOOP():

    def __init__(self, samples_objs, samples_cons, input_dim, grid_size=1000, pareto_set_size=None, feasible_values=0.0, min_distance_between_points=1e-6):

        self.samples_objs = samples_objs
        self.samples_cons = samples_cons
        self.input_dim = input_dim
        self.bounds = [(0.0, 1.0)] * self.input_dim

        self.grid_size = grid_size
        self.pareto_set_size = pareto_set_size

        self.min_distance_between_points = min_distance_between_points
        self.feasible_values = feasible_values

        self.fast_dist = self._dist_einsum if self.input_dim < 10 else self._dist_cdist

    def _dist_einsum(self, x1, x2):
        diff_x1_x2 = x1 - x2[:, None]
        return np.sqrt(np.einsum("ijk,ijk->ij", diff_x1_x2, diff_x1_x2)).squeeze()

    def _dist_cdist(self, x1, x2):
        return cdist(x1, x2)

    def find_feasible_grid(self, constraints, grid, feasible_values=0.0):

        if not isinstance(feasible_values, np.ndarray):
            feasible_values = np.ones(self.input_dim) * feasible_values

        # We obtain the feasible locations of the first constraint

        feasible_region = constraints[ 0 ](grid) >= feasible_values[ 0 ] 

        # We compute the feasible mask of the rest of the constraints and combine all the results

        for i, con_fun in enumerate(constraints[ 1 : ]):
            feasible_region = np.logical_and(feasible_region, con_fun(grid) >= feasible_values[ i + 1 ])

        if not np.any(feasible_region):
            return None
        
        return grid[ feasible_region, : ]
    
    def optimize_obj_globally(self, obj, cons, obj_evals, feasible_grid, constraint_tol=1e-6):

        assert self.input_dim == feasible_grid.shape[ 1 ]

        num_con = len(cons)

        best_guess_index = np.argmin(obj_evals)
        best_guess_value = np.min(obj_evals)

        x_initial = feasible_grid[ best_guess_index, : ]

        f       = lambda x: float(obj(x, gradient=False))
        f_prime = lambda x: obj(x, gradient=True).flatten()

        # with SLSQP in scipy, the constraints are written as c(x) >= 0

        def g(x):
            g_func = np.zeros(num_con)
            for i, constraint_wrapper in enumerate(cons):
                g_func[ i ] = constraint_wrapper(x, gradient=False) - self.feasible_values[ i ]
            return g_func

        def g_prime(x):
            g_grad_func = np.zeros((num_con, self.input_dim))
            for i, constraint_wrapper in enumerate(cons):
                g_grad_func[ i, : ] = constraint_wrapper(x, gradient=True)
            return g_grad_func

        opt_x = spo.fmin_slsqp(f,
                               x_initial.copy(),
                               bounds=self.bounds,
                               disp=0,
                               fprime=f_prime,
                               f_ieqcons=g,
                               fprime_ieqcons=g_prime)

        # We make sure bounds are respected

        opt_x = np.clip(opt_x, a_min=0.0, a_max=1.0)

        if f(opt_x) < best_guess_value and np.all(g(opt_x) >= 0):
            return opt_x[ None ]
        
        # logging.debug('SLSQP failed when optimizing x*')

        # We try to solve the problem again but more carefully (we substract "constraint_tol" to the constraints)

        def g(x):
            g_func = np.zeros(num_con)
            for i,constraint_wrapper in enumerate(cons):
                g_func[ i ] = constraint_wrapper(x, gradient=False) - constraint_tol - self.feasible_values[ i ]
            return g_func

        opt_x = spo.fmin_slsqp(f,
                               x_initial.copy(),
                               bounds=self.bounds,
                               disp=0,
                               fprime=f_prime,
                               f_ieqcons=g,
                               fprime_ieqcons=g_prime)

        opt_x = np.clip(opt_x, a_min=0.0, a_max=1.0)

        if f(opt_x) < best_guess_value and np.all(g(opt_x) >= -constraint_tol):
            return opt_x[ None ]
        
        # logging.debug('SLSQP failed two times when optimizing x*')
        return None

    def _compute_pareto_front(self, pts): # Corresponds to _cull_algorithm() function of Spearmint

        n_points = pts.shape[ 0 ]

        i_pt = 0
        indices_pareto_pts = np.arange(n_points)

        while i_pt < pts.shape[ 0 ]:
            old_i_pt = indices_pareto_pts[ i_pt ]
            
            # We obtain which points are dominated by the current point

            mask_new_pareto = np.any(pts < pts[ i_pt ], axis=1)
            mask_new_pareto[ i_pt ] = True

            # We remove the dominated points

            indices_pareto_pts = indices_pareto_pts[ mask_new_pareto ]
            pts = pts[ mask_new_pareto ]

            # We update the index taking into account the contraction of 'pts'

            i_pt = np.searchsorted(indices_pareto_pts, old_i_pt, side="right")
                
        mask_pareto_front = np.zeros(n_points, dtype = bool)
        mask_pareto_front[ indices_pareto_pts ] = True
        return mask_pareto_front

    def obtain_indices_pareto(self, pts):

        # We sort the pareto front (trick to speed up the algorithm)

        ixs = np.argsort(((pts - pts.mean(axis=0)) / (pts.std(axis=0)+1e-7)).sum(axis=1))
        pts = pts[ ixs ]

        # We obtain the indices of the pareto front given the set 'pts'
        
        mask_pareto_front = self._compute_pareto_front(pts)
        
        # We undo the sorting to return the mask in the same order in which 'pts' was given
        
        mask_pareto_front[ ixs ] = mask_pareto_front.copy()
        
        return mask_pareto_front

    def compute_pareto_front_and_set_summary_y_space(self, pareto_set, pareto_front, pareto_set_size):
        
        assert pareto_set_size > 0

        if pareto_set.shape[ 0 ] <= pareto_set_size:
            return pareto_set, pareto_front

        # We compute the distance of all the points in the function space of the Pareto front between them

        distances = self.fast_dist(pareto_front, pareto_front)

        # Useful hack we add the best observations of each objective first

        subset = np.zeros( pareto_set_size, dtype=np.int64 )

        for i in range(pareto_front.shape[ 1 ]): # We add to the subset the best solution for each objective
            subset[ i ] = np.argmin(pareto_front[ : , i ])

        # Min-max algorithm to choose the next point to include in the summary of the pareto solution

        for n_chosen in range(pareto_front.shape[ 1 ], pareto_set_size):

            # First, we obtain the minimum distance between the candidate points and the points in the current pareto summary

            candidates = subset [ :n_chosen ]

            arr_min_distances = np.min(distances[ candidates, : ], axis=0)

            # Second, we include in the summary the farthest point of the candidates to the summary

            subset[ n_chosen ] = np.argmax(arr_min_distances)

        return pareto_set[ subset, : ], pareto_front[ subset, : ]

    def compute_pareto_solution_from_samples(self, inputs):
        """
        First, we create a grid of candidate points for the pareto set.
        Second, we add the optimum of the objectives to that grid.
        Third, we run the algorithm to obtain the pareto set on the grid.
        Finally, we summarize the pareto solution if requested and return it.
        """

        # Uniform grid with the candidate points for the pareto set

        grid = np.concatenate((np.random.uniform(size=(self.input_dim * self.grid_size, self.input_dim)), inputs))


        # We remove all the infeasible locations of the grid (this speeds up the optimization process)

        if (grid := self.find_feasible_grid(self.samples_cons, grid, feasible_values=self.feasible_values)) is None: # XXX DFS: We use Walrus Operator, python version must be >= 3.8
            return None

        # # We remove the duplicates of the grid
        
        # kdtree = KDTree(grid)
        # ixs_nearby_point_pairs = kdtree.query_pairs(self.min_distance_between_points, output_type='ndarray')
        # if ixs_nearby_point_pairs.shape[ 0 ]:
        #     ixs_duplicates = ixs_nearby_point_pairs[ : , 0 ] # We keep only one of the points from each pair of nearby points
        #     np.delete(grid, ixs_duplicates, 0)

        # We initialize an array of values of the objectives in the grid and 
        # a new array with the location of the optimums of the objectives

        grid_evals = np.empty(shape=(grid.shape[ 0 ], len(self.samples_objs)))
        opt_objs_x = np.array([], dtype=grid.dtype).reshape(0, self.input_dim)
        
        # We get the input of the optimum of each objective

        for i, obj in enumerate(self.samples_objs):

            grid_evals[ : , i ] = obj(grid)

            opt_x = self.optimize_obj_globally(obj, self.samples_cons, grid_evals[ : , i ], grid)

            if ((opt_x is not None) and (np.min(self.fast_dist(grid, opt_x)) > 1e-6)):
                opt_objs_x = np.vstack((opt_objs_x, opt_x))

        # We include in the grid the location of the optimums of the objectives

        if opt_objs_x.shape[ 0 ] > 0:

            grid = np.vstack((grid, opt_objs_x))

            # We include in the array of values of the objectives in the grid the values of the optimums

            opt_objs_y = np.empty(shape=(opt_objs_x.shape[ 0 ], len(self.samples_objs)))

            for i, obj in enumerate(self.samples_objs):

                opt_objs_y[ : , i ] = obj(opt_objs_x)

            grid_evals = np.vstack((grid_evals, opt_objs_y))

        # We optimize the grid

        indices_pareto = self.obtain_indices_pareto(grid_evals)

        pareto_set = grid[ indices_pareto, : ]
        pareto_front = grid_evals[ indices_pareto, : ]

        # We summarize the pareto set

        if self.pareto_set_size is not None:
            pareto_set, pareto_front = self.compute_pareto_front_and_set_summary_y_space(pareto_set, pareto_front, self.pareto_set_size)
        
        return torch.from_numpy(pareto_set), torch.from_numpy(pareto_front), self.samples_objs, self.samples_cons
        
