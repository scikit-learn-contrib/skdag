import numpy as np
from scipy import sparse


def _is_passthrough(estimator):
    return estimator is None or estimator == "passthrough"


def _is_transformer(estimator):
    return (
        hasattr(estimator, "fit") or hasattr(estimator, "fit_transform")
    ) and hasattr(estimator, "transform")


def _is_predictor(estimator):
    return (hasattr(estimator, "fit") or hasattr(estimator, "fit_predict")) and hasattr(
        estimator, "predict"
    )


def _in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def _stack(Xs, axis=0):
    """
    Where an estimator has multiple upstream dependencies, this method defines the
    strategy for merging the upstream outputs into a single input. ``axis=0`` is
    equivalent to a vstack (combination of samples) and ``axis=1`` is equivalent to a
    hstack (combination of features). Higher axes are only supported for non-sparse
    data sources.
    """
    if any(sparse.issparse(f) for f in Xs):
        if -2 <= axis < 2:
            axis = axis % 2
        else:
            raise NotImplementedError(
                f"Stacking is not supported for sparse inputs on axis > 1."
            )

        if axis == 0:
            Xs = sparse.vstack(Xs).tocsr()
        elif axis == 1:
            Xs = sparse.hstack(Xs).tocsr()
    else:
        if axis == 1:
            Xs = np.hstack(Xs)
        else:
            Xs = np.stack(Xs, axis=axis)

    return Xs
