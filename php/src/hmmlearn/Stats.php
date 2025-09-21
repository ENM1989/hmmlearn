<?php

namespace HMM;

/**
 * Functions for calculating log probabilities under multivariate Gaussian distributions.
 *
 * It's assumed that a matrix/numerical library would be used in a real-world PHP scenario.
 * For the purpose of this conversion, we will use basic PHP arrays,
 * and some functions requiring complex matrix operations will be placeholders.
 */
class Stats
{
    const TINY = 1.0e-15; // A small number to prevent log(0) or division by zero.

    /**
     * Compute the log probability under a multivariate Gaussian distribution.
     *
     * @param array $X List of n_features-dimensional data points. Shape: (n_samples, n_features)
     * @param array $means List of mean vectors. Shape: (n_components, n_features)
     * @param mixed $covars Covariance parameters. Shape depends on $covariance_type.
     * @param string $covariance_type The type of covariance. One of: 'spherical', 'tied', 'diag', 'full'.
     * @return array Array of log probabilities. Shape: (n_samples, n_components)
     * @throws \ValueError If covariance_type is invalid.
     */
    public static function log_multivariate_normal_density(array $X, array $means, $covars, string $covariance_type = 'diag'): array
    {
        switch ($covariance_type) {
            case 'spherical':
                return self::_log_multivariate_normal_density_spherical($X, $means, $covars);
            case 'tied':
                return self::_log_multivariate_normal_density_tied($X, $means, $covars);
            case 'diag':
                return self::_log_multivariate_normal_density_diag($X, $means, $covars);
            case 'full':
                return self::_log_multivariate_normal_density_full($X, $means, $covars);
            default:
                throw new \ValueError("Invalid covariance_type: {$covariance_type}");
        }
    }

    /**
     * Compute Gaussian log-density at X for a diagonal model.
     */
    private static function _log_multivariate_normal_density_diag(array $X, array $means, array $covars): array
    {
        $ns = count($X);
        if ($ns === 0) {
            return [];
        }
        $nc = count($means);
        $nf = count($means[0]);

        // Add a tiny epsilon to covars to avoid log(0) or division by zero
        $safe_covars = array_map(function ($row) {
            return array_map(function ($c) {
                return max($c, self::TINY);
            }, $row);
        }, $covars);

        $log_covars_sum = [];
        for ($c = 0; $c < $nc; $c++) {
            $log_covars_sum[$c] = array_sum(array_map('log', $safe_covars[$c]));
        }

        $lpr = array_fill(0, $ns, array_fill(0, $nc, 0.0));

        for ($s = 0; $s < $ns; $s++) {
            for ($c = 0; $c < $nc; $c++) {
                $sum_sq_diff = 0;
                for ($f = 0; $f < $nf; $f++) {
                    $diff = $X[$s][$f] - $means[$c][$f];
                    $sum_sq_diff += ($diff * $diff) / $safe_covars[$c][$f];
                }
                $lpr[$s][$c] = -0.5 * ($nf * log(2 * M_PI) + $log_covars_sum[$c] + $sum_sq_diff);
            }
        }

        return $lpr;
    }

    /**
     * Compute Gaussian log-density at X for a spherical model.
     */
    private static function _log_multivariate_normal_density_spherical(array $X, array $means, array $covars): array
    {
        $nc = count($means);
        $nf = count($means[0]);

        // Broadcast covars to the shape (n_components, n_features)
        $broadcasted_covars = [];
        for ($c = 0; $c < $nc; $c++) {
            $broadcasted_covars[$c] = array_fill(0, $nf, $covars[$c]);
        }

        return self::_log_multivariate_normal_density_diag($X, $means, $broadcasted_covars);
    }

    /**
     * Compute Gaussian log-density at X for a tied model.
     */
    private static function _log_multivariate_normal_density_tied(array $X, array $means, array $covars): array
    {
        $nc = count($means);
        // Broadcast covars to shape (n_components, n_features, n_features)
        $broadcasted_covars = array_fill(0, $nc, $covars);
        return self::_log_multivariate_normal_density_full($X, $means, $broadcasted_covars);
    }

    /**
     * Log probability for full covariance matrices.
     *
     * @todo This is a placeholder. A robust implementation requires a numerical library
     *       for Cholesky decomposition and solving triangular systems.
     */
    private static function _log_multivariate_normal_density_full(array $X, array $means, array $covars): array
    {
        trigger_error(
            "_log_multivariate_normal_density_full() is a placeholder. " .
            "This function requires a numerical library for Cholesky decomposition.",
            E_USER_WARNING
        );

        $ns = count($X);
        $nc = count($means);
        // Return an array of NaNs or zeros to match the expected output shape.
        return array_fill(0, $ns, array_fill(0, $nc, NAN));
    }
}
