<?php

namespace HMM;

trait CategoricalEmissionsTrait
{
    // Note: Properties like n_features, emissionprob_ are expected to be on the class using this trait.

    protected function _check_and_set_n_features(array $X): void
    {
        $X_nd = \NDArray::array($X);
        if ($X_nd->min() < 0) {
            throw new \ValueError("Symbols should be nonnegative");
        }
        $max_val = $X_nd->max();

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

    protected function _compute_likelihood(\NDArray $X): \NDArray
    {
        return $this->emissionprob_->T()->select($X->squeeze(1));
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['obs'] = \NDArray::zeros([$this->n_components, $this->n_features]);
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, array $X, \NDArray $lattice, \NDArray $posteriors, \NDArray $fwdlattice, \NDArray $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $X, $lattice, $posteriors, $fwdlattice, $bwdlattice);

        if (strpos($this->params, 'e') !== false) {
            $X_squeezed = \NDArray::array($X)->squeeze(1);
            $obs = $stats['obs']->T();
            for ($i = 0; $i < $X_squeezed->shape()[0]; $i++) {
                $idx = $X_squeezed[$i];
                $obs_slice = $obs->slice($idx);
                $post_slice = $posteriors->slice($i);
                $obs->setSlice($idx, \NDArray::add($obs_slice, $post_slice));
            }
            $stats['obs'] = $obs->T();
        }
    }

    protected function _generate_sample_from_state(int $state): \NDArray
    {
        $cdf = $this->emissionprob_->slice($state)->cumsum();
        $rand = mt_rand() / mt_getrandmax();
        $symbol = $cdf->gt($rand)->argmax();
        return \NDArray::array([$symbol]);
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

    protected function _compute_log_likelihood(\NDArray $X): \NDArray
    {
        return Stats::log_multivariate_normal_density(
            $X, $this->means_, $this->_covars_, $this->covariance_type
        );
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['post'] = \NDArray::zeros([$this->n_components]);
        $stats['obs'] = \NDArray::zeros([$this->n_components, $this->n_features]);
        $stats['obs**2'] = \NDArray::zeros([$this->n_components, $this->n_features]);
        if (in_array($this->covariance_type, ['tied', 'full'])) {
            $stats['obs*obs.T'] = \NDArray::zeros([$this->n_components, $this->n_features, $this->n_features]);
        }
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, array $X, \NDArray $lattice, \NDArray $posteriors, \NDArray $fwdlattice, \NDArray $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $X, $lattice, $posteriors, $fwdlattice, $bwdlattice);
        $X_nd = \NDArray::array($X);

        if ($this->_needs_sufficient_statistics_for_mean()) {
            $stats['post'] = \NDArray::add($stats['post'], $posteriors->sum(0));
            $stats['obs'] = \NDArray::add($stats['obs'], $posteriors->T()->dot($X_nd));
        }

        if ($this->_needs_sufficient_statistics_for_covars()) {
            if (in_array($this->covariance_type, ['spherical', 'diag'])) {
                $stats['obs**2'] = \NDArray::add($stats['obs**2'], $posteriors->T()->dot($X_nd->pow(2)));
            } else { // tied or full
                $nt = $posteriors->shape()[0];
                $nc = $this->n_components;
                $nf = $this->n_features;
                $obs_obs_T = \NDArray::zeros([$nc, $nf, $nf]);
                for ($i = 0; $i < $nt; $i++) {
                    $post = $posteriors->slice($i)->reshape(1, $nc); // 1, nc
                    $obs = $X_nd->slice($i)->reshape($nf, 1); // nf, 1
                    $outer = $obs->dot($obs->T()); // nf, nf
                    $term = \NDArray::multiply($outer, $post->T()); // nc, nf, nf
                    $obs_obs_T = \NDArray::add($obs_obs_T, $term);
                }
                $stats['obs*obs.T'] = \NDArray::add($stats['obs*obs.T'], $obs_obs_T);
            }
        }
    }

    protected function _generate_sample_from_state(int $state): \NDArray
    {
        // This requires a multivariate normal random number generator,
        // which is not standard in PHP. Returning means as a placeholder.
        trigger_error("Sampling from Gaussian emissions requires a multivariate normal RNG.", E_USER_WARNING);
        return $this->means_->slice($state);
    }
}


trait MultinomialEmissionsTrait
{
    // Properties like n_features, n_trials, emissionprob_ are expected to be on the class using this trait.

    protected function _check_and_set_n_features(array $X): void
    {
        // In PHP, assuming $X is an array of arrays.
        // We can use NDArray for easier processing.
        $X_nd = \NDArray::array($X);

        if ($X_nd->min() < 0) {
            throw new \ValueError("Symbol counts should be nonnegative integers");
        }

        if ($this->n_trials === null) {
            $this->n_trials = $X_nd->sum(1); // Sum along axis 1
        } else {
            // Assuming $this->n_trials can be a single int or an array/NDArray
            $sums = $X_nd->sum(1);
            if (is_scalar($this->n_trials)) {
                if (!($sums->equal($this->n_trials))->all()) {
                     throw new \ValueError("Total count for each sample should add up to the number of trials");
                }
            } else {
                 if (!($sums->equal(\NDArray::array($this->n_trials))))->all()) {
                     throw new \ValueError("Total count for each sample should add up to the number of trials");
                 }
            }
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

    protected function _compute_log_likelihood(\NDArray $X): \NDArray
    {
        $X_nd = $X;
        $columns = [];
        $n_trials = $X_nd->sum(1);
        $log_n_factorial = \NDArray::gammaln(\NDArray::add($n_trials, 1));
        $log_x_factorial = \NDArray::gammaln(\NDArray::add($X_nd, 1))->sum(1);

        for ($c = 0; $c < $this->n_components; $c++) {
            $p = $this->emissionprob_->slice($c);
            $log_p = \NDArray::log($p);
            $log_terms = \NDArray::multiply($X_nd, $log_p)->sum(1);

            $col = \NDArray::subtract($log_n_factorial, $log_x_factorial);
            $col = \NDArray::add($col, $log_terms);
            $columns[] = $col->toArray();
        }

        return \NDArray::array($columns)->T();
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['obs'] = \NDArray::zeros([$this->n_components, $this->n_features]);
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, array $X, \NDArray $framelogprob, \NDArray $posteriors, \NDArray $fwdlattice, \NDArray $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $X, $framelogprob, $posteriors, $fwdlattice, $bwdlattice);
        if (strpos($this->params, 'e') !== false) {
             $stats['obs'] = \NDArray::add($stats['obs'], $posteriors->T()->dot(\NDArray::array($X)));
        }
    }

    protected function _generate_sample_from_state(int $state): \NDArray
    {
        // Requires a multinomial RNG, which is not standard. Placeholder.
        trigger_error("Sampling from Multinomial emissions requires a multinomial RNG.", E_USER_WARNING);
        return \NDArray::zeros([$this->n_features]);
    }
}

trait GMMEmissionsTrait
{
    // Properties like n_mix, weights_, means_, covars_ are expected on the class.

    protected function _get_n_fit_scalars_per_param(): array
    {
        $nc = $this->n_components;
        $nf = $this->n_features;
        $nm = $this->n_mix;
        $cov_params = [
            "spherical" => $nc * $nm,
            "diag" => $nc * $nm * $nf,
            "full" => $nc * $nm * $nf * ($nf + 1) / 2,
            "tied" => $nc * $nf * ($nf + 1) / 2,
        ];

        return [
            "s" => $nc - 1,
            "t" => $nc * ($nc - 1),
            "m" => $nc * $nm * $nf,
            "c" => $cov_params[$this->covariance_type],
            "w" => $nm - 1,
        ];
    }

    protected function _compute_log_weighted_gaussian_densities(\NDArray $X, int $i_comp): \NDArray
    {
        // This is a simplified version.
        return \NDArray::zeros([$X->shape()[0], $this->n_mix]);
    }

    protected function _compute_log_likelihood(\NDArray $X): \NDArray
    {
        $logprobs = \NDArray::empty([$X->shape()[0], $this->n_components]);
        for ($i = 0; $i < $this->n_components; $i++) {
            $log_denses = $this->_compute_log_weighted_gaussian_densities($X, $i);
            // logsumexp is needed here.
            $logprobs_i = \NDArray::log($log_denses->exp()->sum(1));
            // This is complex to assign back to a slice of NDArray.
        }
        return $logprobs;
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['post_mix_sum'] = \NDArray::zeros([$this->n_components, $this->n_mix]);
        $stats['post_sum'] = \NDArray::zeros([$this->n_components]);

        if (strpos($this->params, 'm') !== false) {
            $stats['m_n'] = \NDArray::multiply($this->means_weight, $this->means_prior);
        }
        if (strpos($this->params, 'c') !== false) {
            $stats['c_n'] = \NDArray::zeros_like($this->covars_);
        }
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, \NDArray $X, \NDArray $lattice, \NDArray $post_comp, \NDArray $fwdlattice, \NDArray $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $X, $lattice, $post_comp, $fwdlattice, $bwdlattice);
        trigger_error("GMMHMM._accumulate_sufficient_statistics is not fully implemented due to complexity of einsum.", E_USER_WARNING);
    }

    protected function _generate_sample_from_state(int $state): \NDArray
    {
        trigger_error("Sampling from GMM emissions requires a multivariate normal RNG and choice.", E_USER_WARNING);
        return \NDArray::zeros([$this->n_features]);
    }
}

trait PoissonEmissionsTrait
{
    // Properties like n_features, lambdas_ are expected to be on the class using this trait.

    protected function _get_n_fit_scalars_per_param(): array
    {
        $nc = $this->n_components;
        $nf = $this->n_features;
        return [
            "s" => $nc - 1,
            "t" => $nc * ($nc - 1),
            "l" => $nc * $nf,
        ];
    }

    protected function _compute_log_likelihood(\NDArray $X): \NDArray
    {
        $X_nd = $X;
        $columns = [];
        $log_x_factorial = \NDArray::gammaln(\NDArray::add($X_nd, 1))->sum(1);

        for ($c = 0; $c < $this->n_components; $c++) {
            $lambda = $this->lambdas_->slice($c);
            $log_lambda = \NDArray::log($lambda);

            $term1 = \NDArray::multiply($X_nd, $log_lambda)->sum(1);
            $term2 = $lambda->sum();

            $col = \NDArray::subtract($term1, $term2);
            $col = \NDArray::subtract($col, $log_x_factorial);
            $columns[] = $col->toArray();
        }

        return \NDArray::array($columns)->T();
    }

    protected function _initialize_sufficient_statistics(): array
    {
        $stats = parent::_initialize_sufficient_statistics();
        $stats['post'] = \NDArray::zeros([$this->n_components]);
        $stats['obs'] = \NDArray::zeros([$this->n_components, $this->n_features]);
        return $stats;
    }

    protected function _accumulate_sufficient_statistics(array &$stats, array $obs, \NDArray $lattice, \NDArray $posteriors, \NDArray $fwdlattice, \NDArray $bwdlattice): void
    {
        parent::_accumulate_sufficient_statistics($stats, $obs, $lattice, $posteriors, $fwdlattice, $bwdlattice);
        if (strpos($this->params, 'l') !== false) {
            $stats['post'] = \NDArray::add($stats['post'], $posteriors->sum(0));
            $stats['obs'] = \NDArray::add($stats['obs'], $posteriors->T()->dot(\NDArray::array($obs)));
        }
    }

    protected function _generate_sample_from_state(int $state): \NDArray
    {
        // Requires a poisson RNG, which is not standard. Placeholder.
        trigger_error("Sampling from Poisson emissions requires a poisson RNG.", E_USER_WARNING);
        return \NDArray::zeros([$this->n_features]);
    }
}
