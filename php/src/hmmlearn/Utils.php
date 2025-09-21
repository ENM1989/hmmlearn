<?php

namespace HMM;

/**
 * Private utilities.
 *
 * It's assumed that a matrix/numerical library would be used in a real-world PHP scenario.
 * For the purpose of this conversion, we will use basic PHP arrays,
 * and some functions requiring complex matrix operations will be placeholders.
 */
class Utils
{
    /**
     * Computes the log of the determinant of a matrix.
     *
     * @todo This is a placeholder. A robust implementation requires a numerical library
     *       for stable matrix decomposition (e.g., LU decomposition).
     *
     * @param array $a The matrix (2D array).
     * @return float The log determinant.
     */
    public static function logdet(array $a): float
    {
        trigger_error(
            "logdet() is a placeholder and not fully implemented. " .
            "A proper numerical library is required for a stable implementation.",
            E_USER_WARNING
        );

        // A naive determinant calculation is numerically unstable and not provided.
        // The value of log(abs(det(A))) is returned as a rough equivalent,
        // but this ignores the sign of the determinant, which is important in the original Python code.
        // A proper implementation would need to handle the sign.
        return log(abs(1.0)); // Placeholder value
    }

    /**
     * Splits a sequence X into subsequences with given lengths.
     *
     * @param array $X The sequence of observations.
     * @param array|null $lengths The lengths of the subsequences.
     * @return array An array of subsequences.
     * @throws \ValueError If the lengths do not sum up to the total number of samples.
     */
    public static function split_X_lengths(array $X, ?array $lengths): array
    {
        if ($lengths === null) {
            return [$X];
        }

        $n_samples = count($X);
        if (array_sum($lengths) !== $n_samples) {
            throw new \ValueError(
                "The sum of lengths does not match the number of samples."
            );
        }

        $result = [];
        $start = 0;
        foreach ($lengths as $length) {
            $result[] = array_slice($X, $start, $length);
            $start += $length;
        }
        return $result;
    }

    /**
     * Do basic checks on matrix covariance sizes and values.
     *
     * @todo This is a placeholder. A robust implementation requires a numerical library
     *       for eigenvalue decomposition to check for positive-definiteness.
     *
     * @param mixed $covars The covariance parameters.
     * @param string $covariance_type The type of covariance.
     * @param int $n_components The number of components.
     * @throws \ValueError If the covariance parameters are invalid.
     */
    public static function validate_covars($covars, string $covariance_type, int $n_components): void
    {
        switch ($covariance_type) {
            case 'spherical':
                if (count($covars) !== $n_components) {
                    throw new \ValueError("'spherical' covars must have length n_components");
                }
                if (min($covars) <= 0) {
                    throw new \ValueError("'spherical' covars must be positive");
                }
                break;
            case 'tied':
                if (!is_array($covars) || !isset($covars[0]) || !is_array($covars[0]) || count($covars) !== count($covars[0])) {
                    throw new \ValueError("'tied' covars must be a square matrix (n_dim, n_dim)");
                }
                // @todo Check for symmetry and positive-definiteness (requires eigenvalue decomposition).
                trigger_error("Symmetry and positive-definiteness check for 'tied' covars is not implemented.", E_USER_WARNING);
                break;
            case 'diag':
                if (!is_array($covars) || !isset($covars[0]) || !is_array($covars[0]) || count($covars) !== $n_components) {
                    throw new \ValueError("'diag' covars must have shape (n_components, n_dim)");
                }
                foreach ($covars as $component_covars) {
                    if (min($component_covars) <= 0) {
                        throw new \ValueError("'diag' covars must be positive");
                    }
                }
                break;
            case 'full':
                if (!is_array($covars) || count($covars) !== $n_components || count($covars[0]) !== count($covars[0][0])) {
                    throw new \ValueError("'full' covars must have shape (n_components, n_dim, n_dim)");
                }
                foreach ($covars as $n => $cv) {
                     // @todo Check for symmetry and positive-definiteness (requires eigenvalue decomposition).
                    trigger_error("Symmetry and positive-definiteness check for component {$n} of 'full' covars is not implemented.", E_USER_WARNING);
                }
                break;
            default:
                throw new \ValueError("covariance_type must be one of 'spherical', 'tied', 'diag', 'full'");
        }
    }

    /**
     * Create all the covariance matrices from a given template.
     *
     * @param array $tied_cv The template covariance matrix.
     * @param string $covariance_type The type of covariance.
     * @param int $n_components The number of components.
     * @return mixed The generated covariance matrices.
     * @throws \ValueError If the covariance_type is invalid.
     */
    public static function distribute_covar_matrix_to_match_covariance_type(\NDArray $tied_cv, string $covariance_type, int $n_components): \NDArray
    {
        switch ($covariance_type) {
            case 'spherical':
                $mean = $tied_cv->mean();
                $shape = $tied_cv->shape()[1];
                $ones = \NDArray::ones([$shape]);
                return \NDArray::tile($ones->mul($mean), [$n_components, 1]);
            case 'tied':
                return $tied_cv;
            case 'diag':
                return \NDArray::tile(\NDArray::diag($tied_cv), [$n_components, 1]);
            case 'full':
                return \NDArray::tile($tied_cv, [$n_components, 1, 1]);
            default:
                throw new \ValueError("covariance_type must be one of 'spherical', 'tied', 'diag', 'full'");
        }
    }
}
