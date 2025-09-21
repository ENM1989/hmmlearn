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
    public array $emissionprob_;
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
            $this->emissionprob_ = []; // Simplified random init
            for ($i = 0; $i < $this->n_components; $i++) {
                $row = [];
                $sum = 0;
                for ($j = 0; $j < $this->n_features; $j++) {
                    $val = mt_rand() / mt_getrandmax();
                    $row[] = $val;
                    $sum += $val;
                }
                $this->emissionprob_[] = array_map(fn($v) => $v / $sum, $row);
            }
        }
    }

    protected function _check(): void
    {
        parent::_check();
        if ($this->emissionprob_ === null) {
            throw new \ValueError("emissionprob_ must be initialized");
        }
        if (count($this->emissionprob_) !== $this->n_components || count($this->emissionprob_[0]) !== $this->n_features) {
            throw new \ValueError("emissionprob_ has wrong shape");
        }
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);
        if (strpos($this->params, 'e') !== false) {
            $this->emissionprob_ = $stats['obs'];
            // Normalize
            for ($i = 0; $i < $this->n_components; $i++) {
                $sum = array_sum($this->emissionprob_[$i]);
                if ($sum > 0) {
                    $this->emissionprob_[$i] = array_map(fn($v) => $v / $sum, $this->emissionprob_[$i]);
                }
            }
        }
    }
}

class GaussianHMM extends BaseHMM
{
    use GaussianEmissionsTrait;

    public string $covariance_type;
    public float $min_covar;
    public array $means_;
    public array $_covars_; // Internal storage
    public int $n_features;

    public function __construct(
        int $n_components = 1, string $covariance_type = 'diag', float $min_covar = 1e-3,
        float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        string $algorithm = "viterbi", int $n_iter = 10, float $tol = 1e-2, bool $verbose = false,
        string $params = "stmc", string $init_params = "stmc", string $implementation = "log"
    ) {
        parent::__construct($n_components, $startprob_prior, $transmat_prior, $algorithm, $n_iter, $tol, $verbose, $params, $init_params, $implementation);
        $this->covariance_type = $covariance_type;
        $this->min_covar = $min_covar;
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        parent::_init($X, $lengths);
        $this->n_features = count($X[0]);

        if (strpos($this->init_params, 'm') !== false) {
            // Simplified init without KMeans: pick random samples as means
            $keys = (array) array_rand($X, $this->n_components);
            $this->means_ = array_map(fn($k) => $X[$k], $keys);
        }

        if (strpos($this->init_params, 'c') !== false) {
            // Simplified covars init
            $n_features = count($X[0]);
            $cv = array_fill(0, $n_features, array_fill(0, $n_features, 0.0));
            for($i=0; $i<$n_features; $i++) $cv[$i][$i] = 1.0;

            $this->_covars_ = Utils::distribute_covar_matrix_to_match_covariance_type(
                $cv, $this->covariance_type, $this->n_components
            );
        }
    }

    protected function _check(): void
    {
        parent::_check();
        // Simplified checks
        if (count($this->means_) !== $this->n_components) {
            throw new \ValueError("means_ has wrong shape");
        }
    }

    protected function _do_mstep(array $stats): void
    {
        parent::_do_mstep($stats);
        trigger_error("M-step for GaussianHMM is not implemented.", E_USER_WARNING);
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

// GMMHMM, MultinomialHMM, and PoissonHMM are omitted for brevity.
