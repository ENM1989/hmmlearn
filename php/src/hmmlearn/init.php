<?php

/**
 * hmmlearn-php
 *
 * This file acts as a simple autoloader for the converted hmmlearn library.
 * Including this file will make all the converted classes and traits available.
 *
 * This is a structural, non-functional conversion of the original Python library.
 * Many core features are stubbed out due to dependencies on Python's
 * numerical and scientific computing libraries (NumPy, SciPy).
 */

define('HMM_PHP_VERSION', '0.1.0-converted');

require_once __DIR__ . '/Utils.php';
require_once __DIR__ . '/Stats.php';
require_once __DIR__ . '/BaseHMM.php';
require_once __DIR__ . '/Emissions.php';
require_once __DIR__ . '/HMM.php';
require_once __DIR__ . '/VHMM.php';
require_once __DIR__ . '/KLDivergence.php';
