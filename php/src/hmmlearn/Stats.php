<?php

namespace HMM;

use NDArray;

class Stats
{
    /**
     * Compute the log probability under a multivariate Gaussian distribution.
     */
    public static function log_multivariate_normal_density(NDArray $X, NDArray $means, NDArray $covars, string $covariance_type = 'diag'): NDArray
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

    private static function _log_multivariate_normal_density_diag(NDArray $X, NDArray $means, NDArray $covars): NDArray
    {
        $nf = $means->shape()[1];
        $safe_covars = NDArray::maximum($covars, NDArray::float_min());

        $log_pi = $nf * log(2 * M_PI);
        $log_det = NDArray::log($safe_covars)->sum(1);

        $diff = NDArray::subtract($X->reshape($X->shape()[0], 1, $nf), $means);
        $mahal = NDArray::divide($diff->pow(2), $safe_covars)->sum(2);

        return NDArray::multiply(
            NDArray::add(NDArray::add($log_pi, $log_det), $mahal),
            -0.5
        );
    }

    private static function _log_multivariate_normal_density_spherical(NDArray $X, NDArray $means, NDArray $covars): NDArray
    {
        $nc = $means->shape()[0];
        $nf = $means->shape()[1];

        if ($covars->ndim() == 1) {
            $covars = $covars->reshape(-1, 1);
        }
        $broadcasted_covars = NDArray::tile($covars, [1, $nf]);

        return self::_log_multivariate_normal_density_diag($X, $means, $broadcasted_covars);
    }

    private static function _log_multivariate_normal_density_tied(NDArray $X, NDArray $means, NDArray $covars): NDArray
    {
        $nc = $means->shape()[0];
        $nf = $means->shape()[1];
        $broadcasted_covars = NDArray::tile($covars, [$nc, 1, 1]);
        return self::_log_multivariate_normal_density_full($X, $means, $broadcasted_covars);
    }

    private static function _log_multivariate_normal_density_full(NDArray $X, NDArray $means, NDArray $covars, float $min_covar = 1.e-7): NDArray
    {
        $nc = $means->shape()[0];
        $nf = $means->shape()[1];
        $log_prob = [];

        for ($c = 0; $c < $nc; $c++) {
            $mu = $means->slice($c);
            $cv = $covars->slice($c);

            try {
                $cv_chol = NDArray::linalg_cholesky($cv);
            } catch (\Exception $e) {
                try {
                    $cv_chol = NDArray::linalg_cholesky(NDArray::add($cv, NDArray::eye($nf)->mul($min_covar)));
                } catch (\Exception $e) {
                    throw new \ValueError("'covars' must be symmetric, positive-definite");
                }
            }

            $cv_log_det = NDArray::diag($cv_chol)->log()->sum() * 2;

            $diff = NDArray::subtract($X, $mu);
            $cv_sol = NDArray::linalg_solve_triangular($cv_chol, $diff->T(), true)->T();

            $term1 = $nf * log(2 * M_PI);
            $term2 = $cv_sol->pow(2)->sum(1);
            $term3 = $cv_log_det;

            $log_prob[] = NDArray::multiply(NDArray::add(NDArray::add($term1, $term2), $term3), -0.5)->toArray();
        }

        return NDArray::array($log_prob)->T();
    }
}
