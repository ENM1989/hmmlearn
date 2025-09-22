<?php

namespace HMM;

use SplQueue;
use Exception;

/**
 * @internal A simple seeded random number generator.
 * This is a basic LCG implementation for reproducibility.
 */
class SimpleRandomState
{
    private int $seed;
    private const A = 1664525;
    private const C = 1013904223;
    private const M = 4294967296; // 2**32

    public function __construct(?int $seed = null)
    {
        if ($seed === null) {
            $seed = random_int(0, self::M - 1);
        }
        $this->seed = $seed;
    }

    private function _next_int(): int
    {
        $this->seed = (self::A * $this->seed + self::C);
        // PHP's % operator handles negative numbers differently than C's,
        // so we need to ensure the result is positive.
        $this->seed = ($this->seed % self::M + self::M) % self::M;
        return $this->seed;
    }

    /**
     * Returns a random float between 0.0 (inclusive) and 1.0 (exclusive).
     */
    public function rand(): float
    {
        return $this->_next_int() / self::M;
    }
}


// ##################################################################
// STUBS AND HELPERS
// ##################################################################

/**
 * @internal This is a stub class to emulate the C extension _hmmc.
 * The core HMM algorithms (forward, backward, viterbi) are NOT implemented.
 * Any method calling this class will not be functional.
 */
class _HMMCEmulator
{
    private static function _get_dummy_data($log_frameprob) {
        $n_samples = count($log_frameprob);
        $n_components = count($log_frameprob[0] ?? []);
        return [$n_samples, $n_components];
    }

    public static function forward_log(array $startprob, array $transmat, array $log_frameprob): array {
        trigger_error("C-extension 'hmmc.forward_log' is not implemented.", E_USER_WARNING);
        list($n_samples, $n_components) = self::_get_dummy_data($log_frameprob);
        return [0.0, array_fill(0, $n_samples, array_fill(0, $n_components, 0.0))];
    }

    public static function backward_log(array $startprob, array $transmat, array $log_frameprob): array {
        trigger_error("C-extension 'hmmc.backward_log' is not implemented.", E_USER_WARNING);
        list($n_samples, $n_components) = self::_get_dummy_data($log_frameprob);
        return array_fill(0, $n_samples, array_fill(0, $n_components, 0.0));
    }

    public static function viterbi(array $startprob, array $transmat, array $log_frameprob): array {
        trigger_error("C-extension 'hmmc.viterbi' is not implemented.", E_USER_WARNING);
        list($n_samples, $n_components) = self::_get_dummy_data($log_frameprob);
        return [0.0, array_fill(0, $n_samples, 0)];
    }

    // Other functions from _hmmc are also stubbed
    public static function forward_scaling(array $startprob, array $transmat, array $frameprob): array {
        trigger_error("C-extension 'hmmc.forward_scaling' is not implemented.", E_USER_WARNING);
        list($n_samples, $n_components) = self::_get_dummy_data($frameprob);
        return [0.0, array_fill(0, $n_samples, array_fill(0, $n_components, 0.0)), []];
    }

    public static function backward_scaling(array $startprob, array $transmat, array $frameprob, array $scaling_factors): array {
        trigger_error("C-extension 'hmmc.backward_scaling' is not implemented.", E_USER_WARNING);
        list($n_samples, $n_components) = self::_get_dummy_data($frameprob);
        return array_fill(0, $n_samples, array_fill(0, $n_components, 0.0));
    }

    public static function compute_log_xi_sum(array $fwdlattice, array $transmat, array $bwdlattice, array $lattice): array {
        trigger_error("C-extension 'hmmc.compute_log_xi_sum' is not implemented.", E_USER_WARNING);
        $n_components = count($transmat);
        return array_fill(0, $n_components, array_fill(0, $n_components, 0.0));
    }
}

/**
 * @internal Helper to mimic sklearn.utils.validation.check_is_fitted
 */
function check_is_fitted($model, string $attribute): void {
    if (!isset($model->{$attribute})) {
        throw new Exception("This instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.");
    }
}

// ##################################################################
// MAIN CLASSES
// ##################################################################

class ConvergenceMonitor
{
    private string $_template = "iter %10d | log_prob %16.8f | delta %+16.8f";
    public float $tol;
    public int $n_iter;
    public bool $verbose;
    public SplQueue $history;
    public int $iter = 0;

    public function __construct(float $tol, int $n_iter, bool $verbose)
    {
        $this->tol = $tol;
        $this->n_iter = $n_iter;
        $this->verbose = $verbose;
        $this->history = new SplQueue();
    }

    public function _reset(): void
    {
        $this->iter = 0;
        while (!$this->history->isEmpty()) {
            $this->history->dequeue();
        }
    }

    public function report(float $log_prob): void
    {
        if ($this->verbose) {
            $delta = !$this->history->isEmpty() ? $log_prob - $this->history->bottom() : NAN;
            printf($this->_template . "\n", $this->iter + 1, $log_prob, $delta);
        }

        $precision = pow(2.220446049250313e-16, 0.5);
        if (!$this->history->isEmpty() && ($log_prob - $this->history->bottom()) < -$precision) {
            $delta = $log_prob - $this->history->bottom();
            trigger_error("Model is not converging. Current: {$log_prob} is not greater than {$this->history->bottom()}. Delta is {$delta}", E_USER_WARNING);
        }

        $this->history->enqueue($log_prob);
        if ($this->history->count() > 2) {
            $this->history->dequeue();
        }
        $this->iter++;
    }

    public function hasConverged(): bool
    {
        return ($this->iter >= $this->n_iter ||
                ($this->history->count() >= 2 &&
                 $this->history[1] - $this->history[0] < $this->tol));
    }
}

abstract class _AbstractHMM
{
    public int $n_components;
    public string $algorithm;
    public int $n_iter;
    public float $tol;
    public bool $verbose;
    public string $params;
    public string $init_params;
    public string $implementation;
    public SimpleRandomState $random_state;

    public function __construct(
        int $n_components, string $algorithm, $random_state, int $n_iter,
        float $tol, bool $verbose, string $params, string $init_params, string $implementation
    ) {
        $this->n_components = $n_components;
        $this->algorithm = $algorithm;
        $this->random_state = Utils::check_random_state($random_state);
        $this->n_iter = $n_iter;
        $this->tol = $tol;
        $this->verbose = $verbose;
        $this->params = $params;
        $this->init_params = $init_params;
        $this->implementation = $implementation;
    }

    public function score(array $X, ?array $lengths = null): float
    {
        list($log_prob, ) = $this->_score($X, $lengths, false);
        return $log_prob;
    }

    public function score_samples(array $X, ?array $lengths = null): array
    {
        return $this->_score($X, $lengths, true);
    }

    private function _score(array $X, ?array $lengths, bool $compute_posteriors): array
    {
        check_is_fitted($this, "startprob_");
        $this->_check();

        if ($this->implementation === 'log') {
            return $this->_score_log($X, $lengths, $compute_posteriors);
        }
        // 'scaling' implementation would go here
        throw new Exception("Only 'log' implementation is supported in this version.");
    }

    private function _score_log(array $X, ?array $lengths, bool $compute_posteriors): array
    {
        $log_prob = 0.0;
        $sub_posteriors = [[]]; // Placeholder for empty array of correct shape
        $sequences = Utils::split_X_lengths($X, $lengths);

        foreach ($sequences as $sub_X) {
            $log_frameprob = $this->_compute_log_likelihood($sub_X);
            list($sub_log_prob, $fwdlattice) = _HMMCEmulator::forward_log($this->startprob_, $this->transmat_, $log_frameprob);
            $log_prob += $sub_log_prob;
            if ($compute_posteriors) {
                $bwdlattice = _HMMCEmulator::backward_log($this->startprob_, $this->transmat_, $log_frameprob);
                // $sub_posteriors[] = $this->_compute_posteriors_log($fwdlattice, $bwdlattice);
            }
        }
        // $posteriors = array_merge(...$sub_posteriors);
        return [$log_prob, []]; // Returning empty posteriors due to stubs
    }

    public function decode(array $X, ?array $lengths = null, ?string $algorithm = null): array
    {
        check_is_fitted($this, "startprob_");
        $this->_check();

        $algorithm = $algorithm ?? $this->algorithm;
        if ($algorithm !== 'viterbi' && $algorithm !== 'map') {
            throw new \ValueError("Unknown decoder {$algorithm}");
        }

        if ($algorithm === 'viterbi') {
            $decoder = fn($sub_X) => $this->_decode_viterbi($sub_X);
        } else {
            $decoder = fn($sub_X) => $this->_decode_map($sub_X);
        }

        $log_prob = 0;
        $sub_state_sequences = [];
        $sequences = Utils::split_X_lengths($X, $lengths);

        foreach ($sequences as $sub_X) {
            list($sub_log_prob, $sub_state_sequence) = $decoder($sub_X);
            $log_prob += $sub_log_prob;
            $sub_state_sequences[] = $sub_state_sequence;
        }

        // $state_sequence = array_merge(...$sub_state_sequences);
        return [$log_prob, []]; // Returning empty sequence due to stubs
    }

    private function _decode_viterbi(array $X): array
    {
        $log_frameprob = $this->_compute_log_likelihood($X);
        return _HMMCEmulator::viterbi($this->startprob_, $this->transmat_, $log_frameprob);
    }

    private function _decode_map(array $X): array
    {
        list(, $posteriors) = $this->score_samples($X);
        // This part is not fully convertible without a working score_samples
        return [0.0, []];
    }

    // ... Other abstract or base methods that would be part of a full implementation ...

    abstract protected function _check(): void;
    abstract protected function _compute_log_likelihood(array $X): array;
    abstract protected function _init(array $X, ?array $lengths = null): void;
}

class BaseHMM extends _AbstractHMM
{
    public array $startprob_;
    public array $transmat_;
    public float $startprob_prior;
    public float $transmat_prior;
    public ConvergenceMonitor $monitor_;

    public function __construct(
        int $n_components = 1, float $startprob_prior = 1.0, float $transmat_prior = 1.0,
        string $algorithm = "viterbi", $random_state = null, int $n_iter = 10, float $tol = 1e-2,
        bool $verbose = false, string $params = "st", string $init_params = "st",
        string $implementation = "log"
    ) {
        parent::__construct(
            $n_components, $algorithm, $random_state, $n_iter, $tol, $verbose, $params,
            $init_params, $implementation
        );
        $this->startprob_prior = $startprob_prior;
        $this->transmat_prior = $transmat_prior;
        $this->monitor_ = new ConvergenceMonitor($this->tol, $this->n_iter, $this->verbose);
    }

    protected function _check(): void
    {
        // This is a simplified check
        if (count($this->startprob_) !== $this->n_components) {
            throw new \ValueError("startprob_ must have length n_components");
        }
        if (abs(array_sum($this->startprob_) - 1.0) > 1e-9) {
            throw new \ValueError("startprob_ must sum to 1");
        }
    }

    protected function _compute_log_likelihood(array $X): array
    {
        // To be implemented by subclasses like GaussianHMM
        throw new Exception("Must be overridden in subclass");
    }

    protected function _init(array $X, ?array $lengths = null): void
    {
        $this->n_features = count($X[0] ?? []);
        $init = 1.0 / $this->n_components;

        if (strpos($this->init_params, 's') !== false) {
            $this->startprob_ = array_fill(0, $this->n_components, $init); // Simplified
        }
        if (strpos($this->init_params, 't') !== false) {
            $this->transmat_ = array_fill(0, $this->n_components, array_fill(0, $this->n_components, $init)); // Simplified
        }
    }

    public function get_stationary_distribution(): array
    {
        trigger_error(
            "get_stationary_distribution() is a placeholder. " .
            "This function requires a numerical library for eigenvalue decomposition.",
            E_USER_WARNING
        );
        return array_fill(0, $this->n_components, 1.0 / $this->n_components);
    }
}

// VariationalBaseHMM is omitted for brevity but would follow a similar pattern of
// converting what's possible and stubbing out what's not (digamma, KL divergence).
