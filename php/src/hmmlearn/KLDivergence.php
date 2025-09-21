<?php

namespace HMM;

require_once 'Utils.php';
require_once 'VHMM.php'; // For the Special class stub

class KLDivergence
{
    /**
     * KL Divergence between two dirichlet distributions.
     * @todo Not implemented due to dependency on gammaln and digamma functions.
     */
    public static function kl_dirichlet(array $q, array $p): float
    {
        trigger_error("kl_dirichlet is not implemented. It requires 'gammaln' and 'digamma' functions.", E_USER_WARNING);
        return 0.0;
    }

    /**
     * KL Divergence between two normal distributions.
     */
    public static function kl_normal_distribution(float $mean_q, float $variance_q, float $mean_p, float $variance_p): float
    {
        $result = (log($variance_p / $variance_q)) / 2
                  + (pow($mean_q - $mean_p, 2) + $variance_q) / (2 * $variance_p)
                  - 0.5;
        // assert result >= 0
        return max(0.0, $result);
    }

    /**
     * KL Divergence of two Multivariate Normal Distributions.
     * @todo Not implemented due to dependency on matrix inverse, trace, and logdet.
     */
    public static function kl_multivariate_normal_distribution(array $mean_q, array $covar_q, array $mean_p, array $covar_p): float
    {
        trigger_error("kl_multivariate_normal_distribution is not implemented. It requires matrix operations (inverse, trace, logdet).", E_USER_WARNING);
        return 0.0;
    }

    /**
     * KL Divergence between two gamma distributions.
     * @todo Not implemented due to dependency on gammaln and digamma functions.
     */
    public static function kl_gamma_distribution(float $b_q, float $c_q, float $b_p, float $c_p): float
    {
        trigger_error("kl_gamma_distribution is not implemented. It requires 'gammaln' and 'digamma' functions.", E_USER_WARNING);
        return 0.0;
    }

    /**
     * KL Divergence between two Wishart Distributions.
     * @todo Not implemented due to dependency on matrix operations and special math functions.
     */
    public static function kl_wishart_distribution(float $dof_q, array $scale_q, float $dof_p, array $scale_p): float
    {
        trigger_error("kl_wishart_distribution is not implemented. It requires matrix operations and special math functions.", E_USER_WARNING);
        return 0.0;
    }

    /**
     * Helper for Wishart KL-divergence.
     * @todo Not implemented.
     */
    private static function _E(float $dof, array $scale): float
    {
        trigger_error("_E (helper for Wishart KL) is not implemented.", E_USER_WARNING);
        return 0.0;
    }

    /**
     * Helper for Wishart KL-divergence.
     * @todo Not implemented.
     */
    private static function _logZ(float $dof, array $scale): float
    {
        trigger_error("_logZ (helper for Wishart KL) is not implemented.", E_USER_WARNING);
        return 0.0;
    }
}
