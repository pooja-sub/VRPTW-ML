import gurobipy as gp
from gurobipy import GRB
from paramsVRP import ParamsVRP
from route import Route
from columnGen import ColumnGeneration
import numpy as np
import copy
import os
import pandas as pd

from feature_extraction import generate_arc_features, features_to_dataframe


class LPRelaxationSolver:
    """
    Pure LP Relaxation Solver for VRPTW.
    
    Solves the LP relaxation of the vehicle routing problem with time windows
    using column generation. No branch-and-bound tree traversal - just solves
    the root node LP relaxation to optimality.
    """
    
    def __init__(self):
        pass

    def _build_instance_dataframe(self, user_param):
        """
        Build a DataFrame compatible with feature_extraction.generate_arc_features.
        """
        # Keep only the original depot (0) and customer nodes.
        # Exclude synthetic end depot at index user_param.nbclients.
        node_count = user_param.nbclients  # indices 0..(nbclients-1)
        data = {
            'CUST NO.': user_param.citieslab[:node_count],
            'XCOORD.': user_param.posx[:node_count],
            'YCOORD.': user_param.posy[:node_count],
            'DEMAND': user_param.d[:node_count],
            'READY TIME': user_param.a[:node_count],
            'DUE DATE': user_param.b[:node_count],
            'SERVICE TIME': user_param.s[:node_count],
        }
        return pd.DataFrame(data)

    def _save_labeled_features(self, user_param, column_gen, dataset_path, output_csv=None):
        """
        Generate and save arc features labeled by arcs appearing in generated routes.
        """
        if output_csv is None:
            instance_base = os.path.splitext(os.path.basename(dataset_path))[0]
            customer_size = max(0, user_param.nbclients - 1)
            output_csv = os.path.join(
                os.path.dirname(__file__),
                f"{instance_base}_labeled_{customer_size}.csv"
            )

        df_params = self._build_instance_dataframe(user_param)
        arc_features = generate_arc_features(df_params)
        raw_used_arcs = column_gen.collect_all_generated_arc_labels()

        # Convert two-depot route indexing (start depot=0, end depot=nbclients)
        # into single-depot indexing for CSV features.
        end_depot = user_param.nbclients
        used_arcs = set()
        for i, j in raw_used_arcs:
            i_norm = 0 if i == end_depot else i
            j_norm = 0 if j == end_depot else j
            if i_norm != j_norm:
                used_arcs.add((i_norm, j_norm))

        df_labeled = features_to_dataframe(arc_features, used_arcs)
        df_labeled.to_csv(output_csv, index=False)

        print(f"[Training Data] Labeled feature file saved: {output_csv}")
        print(f"[Training Data] Rows: {len(df_labeled)} | Used arcs labeled: {sum(df_labeled['label'] == 1) if 'label' in df_labeled.columns else 0}")
        return output_csv

    def solve_lp_relaxation(self, user_param, routes, dataset_path=None, output_csv=None):
        """
        Solve the LP relaxation of VRPTW to optimality using column generation.
        
        This method solves the master problem once to optimality by iteratively
        generating negative reduced cost columns until none exist.
        
        :param user_param: Problem parameters (customers, time windows, vehicle capacity, etc.)
        :param routes: Initial routes (typically one route per customer)
        :return: Tuple of (optimal_lp_value, final_routes)
        """
        print("\n" + "="*70)
        print("LP RELAXATION SOLVER - PURE COLUMN GENERATION")
        print("="*70)
        
        # Initialize column generation
        column_gen = ColumnGeneration(user_param)
        
        # Solve LP relaxation to optimality
        optimal_value, final_routes = column_gen.compute_col_gen(routes)

        # Auto-generate labeled training data for this instance.
        if dataset_path is not None:
            self._save_labeled_features(user_param, column_gen, dataset_path, output_csv)
        
        print("\n" + "="*70)
        print("LP RELAXATION SOLVED TO OPTIMALITY")
        print("="*70)
        
        return optimal_value, final_routes
