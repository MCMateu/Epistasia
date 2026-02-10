##########################################
#          REQUIRED LIBRARIES            #
##########################################

import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Optional, Iterable, Sequence, Literal

##########################################
#               CORE MODULE              #
##########################################

@dataclass(frozen=True)
class Landscape:
    """
    Represents a binary landscape with replicates.

    Each row corresponds to one observed binary state (possibly among 2^N total).

    Attributes
    ----------
    states : np.ndarray
        Binary matrix of shape (M, N) with entries in {0, 1}.
    values : np.ndarray
        Float matrix of shape (M, R) with replicate measurements.
        Missing replicates can be NaN.
    N : int
        Dimensionality of the binary space.
    R : int
        Number of replicate measurements per state.
    order : str
        Ordering of states; currently only 'lex' is supported.
    feature_names : list[str]
        Names of the N binary features (default: x1, x2, ..., xN).
    """
    states: np.ndarray
    values: np.ndarray
    N: int
    R: int
    order: str = "lex"
    feature_names: Optional[list[str]] = None

    def __post_init__(self):
        # Basic validation
        states = np.asarray(self.states)
        values = np.asarray(self.values)

        if states.ndim != 2 or states.shape[1] != self.N:
            raise ValueError(f"states must have shape (M, N={self.N}).")
        if values.ndim != 2 or values.shape[1] != self.R:
            raise ValueError(f"values must have shape (M, R={self.R}).")
        if states.shape[0] != values.shape[0]:
            raise ValueError("states and values must have the same number of rows (M).")

        # Binary validation
        if not np.isin(states, [0, 1]).all():
            raise ValueError("states must contain only 0 or 1.")

        object.__setattr__(self, "states", states.astype(np.int8, copy=False))
        object.__setattr__(self, "values", values.astype(float, copy=False))

        if self.order != "lex":
            raise NotImplementedError("Only 'lex' order is supported for now.")
        
        if self.feature_names is None:
            names = [f"x{i+1}" for i in range(self.N)]
        else:
            if len(self.feature_names) != self.N:
                raise ValueError("feature_names must have length N.")
            names = list(self.feature_names)
        object.__setattr__(self, "feature_names", names)

    #######################################
    #           BASIC PROPERTIES          #
    #######################################
    
    @property
    def M(self) -> int:
           """Number of observed states (rows)."""
           return self.states.shape[0]
       
    # -------- Basic views on replicates --------
    def mean_over_replicates(self) -> np.ndarray:
        """Return (M,) mean per state ignoring NaNs in replicate columns."""
        return np.nanmean(self.values, axis=1)

    def drop_rows_with_any_nan(self) -> "Landscape":
        """Keep only rows with complete replicate vectors (no NaNs)."""
        mask = ~np.isnan(self.values).any(axis=1)
        return Landscape(self.states[mask], self.values[mask], self.N, self.R, self.order,
                 feature_names=self.feature_names)

    def drop_rows_with_all_nan(self) -> "Landscape":
        """Keep only rows that have at least one valid replicate (drop rows where all are NaN)."""
        mask = ~np.isnan(self.values).all(axis=1)
        return Landscape(self.states[mask], self.values[mask], self.N, self.R, self.order,
                 feature_names=self.feature_names)


    def missing_states(self) -> np.ndarray:
        """
        Return the list of binary states (as an array) that are not present 
        among the observed configurations.
    
        Returns
        -------
        missing : np.ndarray
            Array of shape (K, N) containing the unobserved states,
            where K = 2^N - M.
        """
        # All possible states in lexicographic order
        all_states = np.array(
            np.meshgrid(*[[0, 1]] * self.N)
        ).T.reshape(-1, self.N)
    
        # Compare observed states
        observed = set(map(tuple, self.states))
        missing = np.array([s for s in all_states if tuple(s) not in observed])
        return missing
    
    def get_values(self, state, replicates: Optional[Sequence[int]] = None, as_dataframe: bool = False):
        """
        Retrieve replicate values for a given binary state.
    
        Parameters
        ----------
        state : array-like, list, tuple, or int
            Binary configuration (e.g. [1,0,1]) or integer encoding of it.
        replicates : optional sequence of int
            If provided, select only these replicate columns.
            Default = all replicates.
        as_dataframe : bool
            If True, return a DataFrame with labeled columns.
    
        Returns
        -------
        np.ndarray or pd.DataFrame
            Row(s) of replicate values corresponding to the given state.
            Returns empty array if the state is not present.
        """
        import pandas as pd
    
        # Convert integer input to binary vector if needed
        if isinstance(state, int):
            if state < 0 or state >= 2**self.N:
                raise ValueError(f"State index {state} out of range for N={self.N}.")
            state = np.array(list(np.binary_repr(state, width=self.N)), dtype=int)
        else:
            state = np.asarray(state, dtype=int)
    
        # Find matching rows
        matches = np.all(self.states == state, axis=1)
        if not np.any(matches):
            print(f"State {state.tolist()} not found.")
            return np.empty((0, self.R))
    
        vals = self.values[matches]
        if replicates is not None:
            vals = vals[:, replicates]
    
        if as_dataframe:
            cols = [f"rep_{i}" for i in range(vals.shape[1])]
            return pd.DataFrame(vals, columns=cols)
    
        return vals


    def select_replicates(self, idx: Sequence[int] | slice) -> "Landscape":
        """Return a Landscape with selected replicate columns (keeps states)."""
        v = self.values[:, idx]
        if v.ndim == 1:
            v = v[:, None]
        new_R = v.shape[1]
        return Landscape(self.states, v, self.N, new_R, self.order,
                 feature_names=self.feature_names)
    
    def select_states(self, idx: Sequence[int] | slice | np.ndarray) -> "Landscape":
        """
        Return a Landscape with a subset of states (rows).
    
        Parameters
        ----------
        idx : sequence | slice | np.ndarray
            Row selector. Can be:
            - A list/array of integer indices
            - A boolean mask of length M
            - A Python slice
    
        Returns
        -------
        Landscape
            New Landscape with selected rows; N, R, order are preserved.
    
        Raises
        ------
        ValueError
            If a boolean mask has wrong length or invalid dtype.
        IndexError
            If integer indices are out of bounds.
        """
        if isinstance(idx, slice):
            rows = range(*idx.indices(self.M))
            idx_arr = np.fromiter(rows, dtype=int)
        else:
            idx_arr = np.asarray(idx)
    
        if idx_arr.dtype == bool:
            if idx_arr.ndim != 1 or idx_arr.shape[0] != self.M:
                raise ValueError(f"Boolean mask must be 1D with length M={self.M}.")
            mask = idx_arr
            states_sel = self.states[mask]
            values_sel = self.values[mask]
        else:
            # integer indexing (let NumPy raise if out of bounds)
            states_sel = self.states[idx_arr]
            values_sel = self.values[idx_arr]
    
        return Landscape(states_sel, values_sel, self.N, self.R, self.order,
                 feature_names=self.feature_names)


    def select_states_by_pattern(self, pattern: Sequence[Optional[int]]) -> "Landscape":
        """
        Select states matching a 0/1/None pattern over the N binary features.
    
        Example
        -------
        pattern = [1, None, 0] selects states with g0==1, g2==0, ignoring g1.
    
        Parameters
        ----------
        pattern : sequence of length N
            Each entry must be 0, 1, or None (None means 'don't care').
    
        Returns
        -------
        Landscape
            New Landscape with rows that match the pattern.
    
        Raises
        ------
        ValueError
            If pattern length != N or contains invalid entries.
        """
        if len(pattern) != self.N:
            raise ValueError(f"pattern must have length N={self.N}.")
        mask = np.ones(self.M, dtype=bool)
        for j, val in enumerate(pattern):
            if val is None:
                continue
            if val not in (0, 1):
                raise ValueError("pattern entries must be 0, 1 or None.")
            mask &= (self.states[:, j] == val)
        return self.select_states(mask)

    #######################################
    #        DataFrame constructor        #
    #######################################
    
    @classmethod
    def from_dataframe(
        cls,
        df: "pd.DataFrame",
        N: Optional[int] = None,
        replicate_cols: Optional[Iterable[str]] = None,
        validate_binary: bool = True,
    ) -> "Landscape":
        
        """
        Build a Landscape from a DataFrame where the first N columns are binary states (0/1)
        and the remaining columns are replicate measurements.

        Parameters
        ----------
        df : pandas.DataFrame
            Input table with M rows. Columns = [state_0, ..., state_{N-1}, rep_0, ..., rep_{R-1}]
            unless replicate_cols is provided.
        N : Optional[int]
            If None, infer N by scanning columns from the left until a non-binary column appears.
            If provided, the first N columns will be used as states (must be in {0,1}).
        replicate_cols : Optional[Iterable[str]]
            If provided, use these columns (in given order) as replicate measurements.
            Otherwise, all columns after the first N will be used.
        validate_binary : bool
            If True, check that state columns contain only {0,1}.

        Returns
        -------
        Landscape
        """
        
        import pandas as pd  # local import to keep dependencies explicit
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas.DataFrame")

        # Determine N if needed
        if N is None:
            N = 0
            for col in df.columns:
                vals = pd.unique(df[col].dropna())
                if len(vals) == 0 or set(np.asarray(vals, dtype=float)).issubset({0.0, 1.0}):
                    N += 1
                else:
                    break
            if N == 0:
                raise ValueError("Could not infer N: the first column is not binary.")
        else:
            if N <= 0:
                raise ValueError("N must be a positive integer.")
            if N > df.shape[1]:
                raise ValueError("N exceeds the number of columns in df.")

        # Slice states and replicate matrix
        state_block = df.iloc[:, :N].to_numpy(copy=False)
        feature_names = list(df.columns[:N]) 
        if validate_binary:
            if pd.isna(state_block).any():
                raise ValueError("State columns must not contain NaN.")
            uniq = np.unique(state_block)
            if not np.all(np.isin(uniq, [0, 1])):
                raise ValueError("State columns must be binary {0,1}.")

        if replicate_cols is None:
            if df.shape[1] == N:
                raise ValueError("No replicate columns found after the first N columns.")
            value_block = df.iloc[:, N:].to_numpy(dtype=float, copy=False)
        else:
            value_block = df.loc[:, list(replicate_cols)].to_numpy(dtype=float, copy=False)

        # Ensure 2D
        if value_block.ndim == 1:
            value_block = value_block[:, None]

        R = value_block.shape[1]
        return cls(
            states=state_block.astype(np.int8, copy=False),
            values=value_block.astype(float, copy=False),
            N=N,
            R=R,
            order="lex",
            feature_names=feature_names,       # ← NEW
        )
    
    #########################################
    #          FOR BEATIFUL DISPLAY         #
    #########################################
    
    # --------------------------------------------------------------------------
    # Helper methods for display and indexing
    # --------------------------------------------------------------------------

    @staticmethod
    def _bits_to_str(bits: np.ndarray) -> str:
        """Convert an array of bits, e.g. [0,1,1,0], into a string '0110'."""
        return ''.join(map(str, bits.tolist()))

    def _make_index(
        self,
        mode: Literal["bits", "int", "multi"] = "bits",
        include_hamming: bool = False,
    ):
        """
        Build an index for the DataFrame representation:
        - 'bits'  → string binary index like '0101...'
        - 'int'   → integer index (0..2^N-1)
        - 'multi' → MultiIndex (bits, int[, Hamming weight])
        """
        # Compute integer representation of each binary state
        ints = (self.states * (1 << np.arange(self.N - 1, -1, -1))).sum(axis=1)
        if mode == "bits":
            idx_bits = [self._bits_to_str(row) for row in self.states]
            if include_hamming:
                h = self.states.sum(axis=1)
                if pd is not None:
                    return pd.MultiIndex.from_arrays([idx_bits, h], names=["state", "Order"])
                else:
                    return idx_bits
            return idx_bits
        elif mode == "int":
            return ints
        elif mode == "multi":
            idx_bits = [self._bits_to_str(row) for row in self.states]
            h = self.states.sum(axis=1)
            if pd is not None:
                return pd.MultiIndex.from_arrays([idx_bits, ints, h], names=["state", "id", "Order"])
            else:
                return idx_bits
        else:
            return None

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def to_dataframe(
        self,
        *,
        index: Literal["bits", "int", "multi"] | None = "bits",
        include_states: bool = True,
        include_hamming: bool = False,
        replicate_names: Optional[list[str]] = None,
        copy: bool = False,
    ):
        """
        Convert the Landscape object into a tidy pandas DataFrame.

        By default, the DataFrame includes:
        - one column per binary state (using feature_names)
        - one column per replicate (rep_1, rep_2, ...)

        Parameters
        ----------
        index : {"bits", "int", "multi", None}
            Index type for the DataFrame. If None, use a default RangeIndex.
        include_states : bool
            Whether to include binary state columns explicitly.
        include_hamming : bool
            Whether to include Hamming distance in the index (if applicable).
        replicate_names : list of str, optional
            Column names for replicate values.
        copy : bool
            Whether to copy the underlying data.
        """
        if pd is None:
            raise ImportError("Pandas is required for to_dataframe(). Please install pandas.")

        # Replicate columns
        cols = (
            replicate_names
            if replicate_names is not None
            else [f"rep_{r+1}" for r in range(self.R)]
        )

        values = self.values.copy() if copy else self.values
        df_values = pd.DataFrame(values, columns=cols)

        # Add state columns if requested
        if include_states:
            df_states = pd.DataFrame(
                self.states,
                columns=self.feature_names
            )
            df = pd.concat([df_states, df_values], axis=1)
        else:
            df = df_values

        # Set index (optional)
        if index is not None:
            idx = self._make_index(index, include_hamming=include_hamming)
            df.index = idx

        return df

    # --------------------------------------------------------------------------
    # Pretty display in notebooks and terminal
    # --------------------------------------------------------------------------

    def _repr_html_(self):
        """
        HTML representation for Jupyter notebooks.
        Displays a preview (first 10 rows) of the landscape as a DataFrame.
        """
        if pd is None:
            preview = "\n".join(
                f"{self._bits_to_str(self.states[i])}\t"
                + " ".join(f"{v:.3g}" for v in self.values[i, :])
                for i in range(min(10, self.values.shape[0]))
            )
            return f"<pre>Landscape(N={self.N}, R={self.R})\n{preview}\n...</pre>"
        df = self.to_dataframe(index="bits", include_hamming=True)
        return df.head(10)._repr_html_()

    def __repr__(self):
        """
        Text representation for terminal environments.
        Falls back to a compact DataFrame preview if pandas is available.
        """
        if pd is None:
            return f"Landscape(N={self.N}, R={self.R}, shape={self.values.shape})"
        df = self.to_dataframe(index="bits", include_hamming=True)
        head = df.head(10).to_string()
        return f"Landscape(N={self.N}, R={self.R}, shape={self.values.shape})\n{head}\n..."