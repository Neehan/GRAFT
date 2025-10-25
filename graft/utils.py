"""Common utility functions for GRAFT."""


def get_knn_suffix(config):
    """Generate kNN suffix for filenames based on config.

    Args:
        config: Config dict with data.semantic_k and data.knn_only

    Returns:
        Suffix string (e.g., '_knn_only15', '_knn3', or '')
    """
    semantic_k = config["data"].get("semantic_k")
    if semantic_k is None:
        return ""

    knn_only = config["data"].get("knn_only", False)
    return f"_knn_only{semantic_k}" if knn_only else f"_knn{semantic_k}"
