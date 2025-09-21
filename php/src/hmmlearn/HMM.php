<?php

namespace HMM;

require_once 'BaseHMM.php';
require_once 'Stats.php';
require_once 'Utils.php';
require_once 'Emissions.php';

class CategoricalHMM extends BaseHMM
{
    use CategoricalEmissionsTrait;

    public ?int $n_features;
    public \NDArray $emissionprob_;
    public float $emissionprob_prior;

    public function __construct(
        int $n_components = 1, float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        float $emissionprob_prior = 1.0, ?int $n_features = null, string $algorithm = "viterbi",
        int $n_iter = 10, float $tol = 1e-2, bool $verbose = false,
        string $params = "ste", string $init_params = "ste", string $implementation = "log"
    ) {
        parent::__construct($n_components, $startprob_prior, $transmat_prior, $algorithm, $n_iter, $tol, $verbose, $params, $init_params, $implementation);
        $this->emissionprob_prior = $emissionprob_prior;
        $this->n_features = $n_features;
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        parent::_init($X, $lengths);
        $this->_check_and_set_n_features($X);

        if (strpos($this->init_params, 'e') !== false) {
            $this->emissionprob_ = \NDArray::random_uniform(0, 1, [$this->n_components, $this->n_features]);
            $this->emissionprob_ = $this->emissionprob_->div($this->emissionprob_->sum(1, true));
        }
    }

    protected function _check(): void
    {
        parent::_check();
        if (!isset($this->emissionprob_)) {
            $this->emissionprob_ = \NDArray::empty();
        }
        if ($this->emissionprob_->shape() !== [$this->n_components, $this->n_features]) {
            throw new \ValueError("emissionprob_ has wrong shape");
        }
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);
        if (strpos($this->params, 'e') !== false) {
            $obs = \NDArray::array($stats['obs']);
            $this->emissionprob_ = \NDArray::maximum(0, \NDArray::add($obs, $this->emissionprob_prior - 1));
            $this->emissionprob_ = $this->emissionprob_->div($this->emissionprob_->sum(1, true));
        }
    }
}

class GaussianHMM extends BaseHMM
{
    use GaussianEmissionsTrait;

    public string $covariance_type;
    public float $min_covar;
    public \NDArray $means_;
    public \NDArray $_covars_; // Internal storage
    public int $n_features;
    public $means_prior;
    public $means_weight;
    public $covars_prior;
    public $covars_weight;

    public function __construct(
        int $n_components = 1, string $covariance_type = 'diag', float $min_covar = 1e-3,
        float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        $means_prior = 0, $means_weight = 0, $covars_prior = 1e-2, $covars_weight = 1,
        string $algorithm = "viterbi", int $n_iter = 10, float $tol = 1e-2, bool $verbose = false,
        string $params = "stmc", string $init_params = "stmc", string $implementation = "log"
    ) {
        parent::__construct($n_components, $startprob_prior, $transmat_prior, $algorithm, $n_iter, $tol, $verbose, $params, $init_params, $implementation);
        $this->covariance_type = $covariance_type;
        $this->min_covar = $min_covar;
        $this->means_prior = $means_prior;
        $this->means_weight = $means_weight;
        $this->covars_prior = $covars_prior;
        $this->covars_weight = $covars_weight;
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        parent::_init($X, $lengths);
        $X_nd = \NDArray::array($X);
        $this->n_features = $X_nd->shape()[1];

        if (strpos($this->init_params, 'm') !== false) {
            // Simplified init without KMeans: pick random samples as means
            $keys = array_rand($X, $this->n_components);
            $means_array = array_map(fn($k) => $X[$k], (array) $keys);
            $this->means_ = \NDArray::array($means_array);
        }

        if (strpos($this->init_params, 'c') !== false) {
            $cv = \NDArray::cov($X_nd->T());
            $cv = \NDArray::add($cv, \NDArray::eye($this->n_features)->mul($this->min_covar));

            // This function needs to be implemented in Utils.php
            $this->_covars_ = Utils::distribute_covar_matrix_to_match_covariance_type(
                $cv, $this->covariance_type, $this->n_components
            );
        }
    }

    protected function _check(): void
    {
        parent::_check();
        if (!isset($this->means_)) {
            $this->means_ = \NDArray::empty();
        }
        if ($this->means_->shape() !== [$this->n_components, $this->n_features]) {
            throw new \ValueError("means_ has wrong shape");
        }
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);

        $denom = $stats['post']->reshape(-1, 1);
        if (strpos($this->params, 'm') !== false) {
            $numerator = \NDArray::add($stats['obs'], $this->means_prior * $this->means_weight);
            $denominator = \NDArray::add($denom, $this->means_weight);
            $this->means_ = $numerator->div($denominator);
        }

        if (strpos($this->params, 'c') !== false) {
            $meandiff = \NDArray::subtract($this->means_, $this->means_prior);

            if (in_array($this->covariance_type, ['spherical', 'diag'])) {
                $c_n = \NDArray::add(
                    $stats['obs**2'],
                    \NDArray::multiply($meandiff->pow(2), $this->means_weight)
                );
                $c_n = \NDArray::subtract($c_n, \NDArray::multiply($this->means_, $stats['obs'])->mul(2));
                $c_n = \NDArray::add($c_n, \NDArray::multiply($this->means_->pow(2), $denom));

                $c_d = \NDArray::add($denom, max($this->covars_weight - 1, 0));

                $this->_covars_ = \NDArray::divide(\NDArray::add($this->covars_prior, $c_n), \NDArray::maximum($c_d, 1e-5));

                if ($this->covariance_type == 'spherical') {
                    $this->_covars_ = \NDArray::tile($this->_covars_->mean(1)->reshape(-1, 1), [1, $this->_covars_->shape()[1]]);
                }
            } else {
                // 'tied', 'full' are more complex and require outer products and einsum,
                // which might not be directly available or may have different syntax in NumPower.
                trigger_error("M-step for 'tied' and 'full' covariance types is not fully implemented.", E_USER_WARNING);
            }
        }
    }

    // These are required by the GaussianEmissionsTrait
    protected function _needs_sufficient_statistics_for_mean(): bool
    {
        return strpos($this->params, 'm') !== false;
    }

    protected function _needs_sufficient_statistics_for_covars(): bool
    {
        return strpos($this->params, 'c') !== false;
    }
}

class GMMHMM extends BaseHMM
{
    use GMMEmissionsTrait;

    public int $n_mix;
    public \NDArray $weights_;
    public \NDArray $means_;
    public \NDArray $covars_;
    public $weights_prior;
    public $means_prior;
    public $means_weight;
    public $covars_prior;
    public $covars_weight;
    public string $covariance_type;
    public float $min_covar;

    public function __construct(
        int $n_components = 1, int $n_mix = 1, float $min_covar = 1e-3,
        float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        $weights_prior = 1.0, $means_prior = 0.0, $means_weight = 0.0,
        $covars_prior = null, $covars_weight = null,
        string $algorithm = "viterbi", string $covariance_type = "diag",
        int $n_iter = 10, float $tol = 1e-2, bool $verbose = false,
        string $params = "stmcw", string $init_params = "stmcw", string $implementation = "log"
    ) {
        parent::__construct($n_components, $startprob_prior, $transmat_prior, $algorithm, $n_iter, $tol, $verbose, $params, $init_params, $implementation);
        $this->n_mix = $n_mix;
        $this->min_covar = $min_covar;
        $this->weights_prior = $weights_prior;
        $this->means_prior = $means_prior;
        $this->means_weight = $means_weight;
        $this->covars_prior = $covars_prior;
        $this->covars_weight = $covars_weight;
        $this->covariance_type = $covariance_type;
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        parent::_init($X, $lengths);
        // Simplified init. The Python version uses KMeans, which is complex to replicate here.
        if (strpos($this->init_params, 'w') !== false) {
            $this->weights_ = \NDArray::full([$this->n_components, $this->n_mix], 1.0 / $this->n_mix);
        }
        if (strpos($this->init_params, 'm') !== false) {
            $this->means_ = \NDArray::random_uniform(0, 1, [$this->n_components, $this->n_mix, $this->n_features]);
        }
        if (strpos($this->init_params, 'c') !== false) {
            // Simplified covars init
            $this->covars_ = \NDArray::ones([$this->n_components, $this->n_mix, $this->n_features]);
        }
    }

    protected function _check(): void
    {
        parent::_check();
        // Simplified checks
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);
        if (strpos($this->params, 'w') !== false) {
            $alphas_minus_one = $this->weights_prior - 1;
            $w_n = \NDArray::add($stats['post_mix_sum'], $alphas_minus_one);
            $w_d = \NDArray::add($stats['post_sum'], $alphas_minus_one->sum(1))->reshape(-1, 1);
            $this->weights_ = $w_n->div($w_d);
        }
        if (strpos($this->params, 'm') !== false) {
            $m_n = $stats['m_n'];
            $m_d = \NDArray::add($stats['post_mix_sum'], $this->means_weight);
            // Missing part to handle zero weights
            $this->means_ = $m_n->div($m_d->reshape($m_d->shape()[0], $m_d->shape()[1], 1));
        }
        if (strpos($this->params, 'c') !== false) {
            $lambdas = $this->means_weight;
            $mus = $this->means_prior;
            $centered_means = \NDArray::subtract($this->means_, $mus);

            if ($this->covariance_type == 'diag') {
                $alphas = $this->covars_prior;
                $betas = $this->covars_weight;
                $centered_means2 = $centered_means->pow(2);

                $c_n = \NDArray::multiply($centered_means2, $lambdas->reshape($lambdas->shape()[0], $lambdas->shape()[1], 1));
                $c_n = \NDArray::add($c_n, \NDArray::multiply($betas, 2));
                $c_n = \NDArray::add($c_n, $stats['c_n']);

                $c_d = \NDArray::add($stats['post_mix_sum']->reshape($stats['post_mix_sum']->shape()[0], $stats['post_mix_sum']->shape()[1], 1), 1);
                $c_d = \NDArray::add($c_d, \NDArray::multiply(\NDArray::add($alphas, 1), 2));

                $this->covars_ = $c_n->div($c_d);

            } elseif ($this->covariance_type == 'spherical') {
                $centered_means_norm2 = $centered_means->pow(2)->sum(-1);

                $alphas = $this->covars_prior;
                $betas = $this->covars_weight;

                $c_n = \NDArray::multiply($lambdas, $centered_means_norm2);
                $c_n = \NDArray::add($c_n, \NDArray::multiply($betas, 2));
                $c_n = \NDArray::add($c_n, $stats['c_n']);

                $c_d = \NDArray::multiply(\NDArray::add($stats['post_mix_sum'], 1), $this->n_features);
                $c_d = \NDArray::add($c_d, \NDArray::multiply(\NDArray::add($alphas, 1), 2));

                $this->covars_ = $c_n->div($c_d);

            } else {
                trigger_error("GMMHMM M-step for full and tied covars is not implemented.", E_USER_WARNING);
            }
        }
    }
}

class MultinomialHMM extends BaseHMM
{
    use MultinomialEmissionsTrait;

    public $n_trials;
    public \NDArray $emissionprob_;

    public function __construct(
        int $n_components = 1, $n_trials = null, float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        string $algorithm = "viterbi", int $n_iter = 10, float $tol = 1e-2, bool $verbose = false,
        string $params = "ste", string $init_params = "ste", string $implementation = "log"
    ) {
        parent::__construct($n_components, $startprob_prior, $transmat_prior, $algorithm, $n_iter, $tol, $verbose, $params, $init_params, $implementation);
        $this->n_trials = $n_trials;
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        parent::_init($X, $lengths);
        if (strpos($this->init_params, 'e') !== false) {
            $this->emissionprob_ = \NDArray::random_uniform(0, 1, [$this->n_components, $this->n_features]);
            $this->emissionprob_ = $this->emissionprob_->div($this->emissionprob_->sum(1, true));
        }
    }

    protected function _check(): void
    {
        parent::_check();
        if (!isset($this->emissionprob_)) {
             $this->emissionprob_ = \NDArray::empty();
        }
        $n_features = $this->emissionprob_->shape()[1] ?? null;
        if ($n_features !== null && $this->emissionprob_->shape() !== [$this->n_components, $n_features]) {
            throw new \ValueError("emissionprob_ must have shape (n_components, n_features)");
        } else {
            $this->n_features = $n_features;
        }

        if ($this->n_trials === null) {
            throw new \ValueError("n_trials must be set");
        }
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);
        if (strpos($this->params, 'e') !== false) {
            $this->emissionprob_ = $stats['obs']->div($stats['obs']->sum(1, true));
        }
    }
}

class PoissonHMM extends BaseHMM
{
    use PoissonEmissionsTrait;

    public \NDArray $lambdas_;
    public $lambdas_prior;
    public $lambdas_weight;

    public function __construct(
        int $n_components = 1, float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        $lambdas_prior = 0.0, $lambdas_weight = 0.0,
        string $algorithm = "viterbi", int $n_iter = 10, float $tol = 1e-2, bool $verbose = false,
        string $params = "stl", string $init_params = "stl", string $implementation = "log"
    ) {
        parent::__construct($n_components, $startprob_prior, $transmat_prior, $algorithm, $n_iter, $tol, $verbose, $params, $init_params, $implementation);
        $this->lambdas_prior = $lambdas_prior;
        $this->lambdas_weight = $lambdas_weight;
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        parent::_init($X, $lengths);
        if (strpos($this->init_params, 'l') !== false) {
            $X_nd = \NDArray::array($X);
            $mean_X = $X_nd->mean();
            $var_X = $X_nd->var();
            // Simplified gamma random generation.
            $this->lambdas_ = \NDArray::random_uniform($mean_X - sqrt($var_X), $mean_X + sqrt($var_X), [$this->n_components, $this->n_features]);
        }
    }

    protected function _check(): void
    {
        parent::_check();
        if (!isset($this->lambdas_)) {
            $this->lambdas_ = \NDArray::empty();
        }
        $n_features = $this->lambdas_->shape()[1] ?? null;
        if ($n_features !== null && $this->lambdas_->shape() !== [$this->n_components, $n_features]) {
            throw new \ValueError("lambdas_ must have shape (n_components, n_features)");
        }
        $this->n_features = $n_features;
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);

        if (strpos($this->params, 'l') !== false) {
            $alphas = $this->lambdas_prior;
            $betas = $this->lambdas_weight;
            $n = $stats['post']->sum();
            $y_bar = $stats['obs']->div($stats['post']->reshape(-1, 1));

            $numerator = \NDArray::add($alphas, $y_bar->mul($n));
            $denominator = $betas + $n;

            $this->lambdas_ = $numerator->div($denominator);
        }
    }
}
