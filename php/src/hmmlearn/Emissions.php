<?php

namespace HMM;

trait CategoricalEmissionsTrait
{
    // Note: Properties like n_features, emissionprob_ are expected to be on the class using this trait.

    protected function _check_and_set_n_features(array $X): void
    {
        // Simplified check, assuming X is already integers.
        $max_val = 0;
        foreach ($X as $row) {
            $max_val = max($max_val, max($row));
        }

        if ($this->n_features === null) {
            $this->n_features = $max_val + 1;
        } elseif ($this->n_features - 1 < $max_val) {
            throw new \ValueError("Largest symbol is {$max_val} but model only emits up to " . ($this->n_features - 1));
        }
    }

    protected function _get_n_fit_scalars_per_param(): array
    {
        $nc = $this->n_components;
        $nf = $this->n_features;
        return [
            "s" => $nc - 1,
            "t" => $nc * ($nc - 1),
            "e" => $nc * ($nf - 1),
        ];
    }

    protected function _compute_likelihood(array $X): array
    {
        $frameprob = [];
        foreach ($X as $sample) {
            $symbol = $sample[0];
            $row = [];
            for ($i = 0; $i < $this->n_components; $i++) {
                $row[] = $this->emissionprob_[$i][$symbol];
            }
            $frameprob[] = $row;
        }
        return $frameprob;
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['obs'] = array_fill(0, $this->n_components, array_fill(0, $this->n_features, 0.0));
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, array $X, array $lattice, array $posteriors, array $fwdlattice, array $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $X, $lattice, $posteriors, $fwdlattice, $bwdlattice);

        if (strpos($this->params, 'e') !== false) {
            for ($t = 0; $t < count($X); $t++) {
                $symbol = $X[$t][0];
                for ($j = 0; $j < $this->n_components; $j++) {
                    $stats['obs'][$j][$symbol] += $posteriors[$t][$j];
                }
            }
        }
    }

    protected function _generate_sample_from_state(int $state): array
    {
        $cdf = [];
        $sum = 0;
        foreach ($this->emissionprob_[$state] as $p) {
            $sum += $p;
            $cdf[] = $sum;
        }

        $rand = mt_rand() / mt_getrandmax();
        foreach ($cdf as $i => $p) {
            if ($rand < $p) {
                return [$i];
            }
        }
        return [count($cdf) - 1];
    }
}


trait GaussianEmissionsTrait
{
    // Properties like means_, _covars_, covariance_type are expected on the class.

    abstract protected function _needs_sufficient_statistics_for_mean(): bool;
    abstract protected function _needs_sufficient_statistics_for_covars(): bool;

    protected function _get_n_fit_scalars_per_param(): array
    {
        $nc = $this->n_components;
        $nf = $this->n_features;
        $cov_params = [
            "spherical" => $nc,
            "diag" => $nc * $nf,
            "full" => $nc * $nf * ($nf + 1) / 2,
            "tied" => $nf * ($nf + 1) / 2,
        ];
        return [
            "s" => $nc - 1,
            "t" => $nc * ($nc - 1),
            "m" => $nc * $nf,
            "c" => $cov_params[$this->covariance_type],
        ];
    }

    protected function _compute_log_likelihood(array $X): array
    {
        return Stats::log_multivariate_normal_density(
            $X, $this->means_, $this->_covars_, $this->covariance_type
        );
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['post'] = array_fill(0, $this->n_components, 0.0);
        $stats['obs'] = array_fill(0, $this->n_components, array_fill(0, $this->n_features, 0.0));
        $stats['obs**2'] = array_fill(0, $this->n_components, array_fill(0, $this->n_features, 0.0));
        if (in_array($this->covariance_type, ['tied', 'full'])) {
            $stats['obs*obs.T'] = array_fill(0, $this->n_components, array_fill(0, $this->n_features, array_fill(0, $this->n_features, 0.0)));
        }
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, array $X, array $lattice, array $posteriors, array $fwdlattice, array $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $X, $lattice, $posteriors, $fwdlattice, $bwdlattice);

        if ($this->_needs_sufficient_statistics_for_mean()) {
            for ($j = 0; $j < $this->n_components; $j++) {
                $post_sum = 0;
                for ($t = 0; $t < count($X); $t++) {
                    $post_sum += $posteriors[$t][$j];
                }
                $stats['post'][$j] += $post_sum;
            }
            // obs += posteriors.T @ X
            for ($j = 0; $j < $this->n_components; $j++) {
                for ($f = 0; $f < $this->n_features; $f++) {
                    $obs_sum = 0;
                    for ($t = 0; $t < count($X); $t++) {
                        $obs_sum += $posteriors[$t][$j] * $X[$t][$f];
                    }
                    $stats['obs'][$j][$f] += $obs_sum;
                }
            }
        }

        if ($this->_needs_sufficient_statistics_for_covars()) {
            // This is a simplified version. A full conversion of the einsum
            // operation for 'full' and 'tied' covariances is very complex.
            trigger_error("Sufficient statistics for 'full' and 'tied' covariances are not fully implemented.", E_USER_WARNING);
        }
    }

    protected function _generate_sample_from_state(int $state): array
    {
        // This requires a multivariate normal random number generator,
        // which is not standard in PHP. Returning means as a placeholder.
        trigger_error("Sampling from Gaussian emissions requires a multivariate normal RNG.", E_USER_WARNING);
        return $this->means_[$state];
    }
}

// Traits for GMM, Multinomial, and Poisson emissions would follow a similar pattern.
// They are omitted for brevity.
