import math
import multiprocessing
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from loguru import logger

from tiberate.config.security_parameters import maximum_qbits

CACHE_FOLDER = os.path.dirname(__file__)


def MillerRabinPrimalityTest(number, rounds=10):
    # If the input is an even number, return immediately with False.
    if number == 2:
        return True
    elif number == 1 or number % 2 == 0:
        return False

    # First we want to express n as : 2^s * r ( were r is odd )

    # The odd part of the number
    oddPartOfNumber = number - 1

    # The number of time that the number is divided by two
    timesTwoDividNumber = 0

    # while r is even divid by 2 to find the odd part
    while oddPartOfNumber % 2 == 0:
        oddPartOfNumber = oddPartOfNumber / 2
        timesTwoDividNumber = timesTwoDividNumber + 1

    # Make oddPartOfNumber integer.
    oddPartOfNumber = int(oddPartOfNumber)

    # Since there are number that are cases of "strong liar" we need to check more than one number
    for time in range(rounds):
        # Choose "Good" random number
        while True:
            # Draw a RANDOM number in range of number ( Z_number )
            randomNumber = random.randint(2, number) - 1
            if randomNumber != 0 and randomNumber != 1:
                break

        # randomNumberWithPower = randomNumber^oddPartOfNumber mod number
        randomNumberWithPower = pow(randomNumber, oddPartOfNumber, number)

        # If random number is not 1 and not -1 ( in mod n )
        if (randomNumberWithPower != 1) and (
            randomNumberWithPower != number - 1
        ):
            # number of iteration
            iterationNumber = 1

            # While we can squre the number and the squered number is not -1 mod number
            while (iterationNumber <= timesTwoDividNumber - 1) and (
                randomNumberWithPower != number - 1
            ):
                # Squre the number
                randomNumberWithPower = pow(randomNumberWithPower, 2, number)

                # inc the number of iteration
                iterationNumber = iterationNumber + 1

            # If x != -1 mod number then it is because we did not find strong witnesses
            # hence 1 have more then two roots in mod n ==>
            # n is composite ==> return false for primality

            if randomNumberWithPower != (number - 1):
                return False

    # The number pass the tests ==> it is probably prime ==> return true for primality
    return True


def generate_N_M(logN: list[int] | None = None):
    # default to generate logN from 12 to 18
    logN = list(range(12, 18)) if logN is None else logN
    logN = sorted(set(logN))  # sort logN and make it unique

    filename = "N_M_" + "_".join([str(lN) for lN in logN]) + ".pkl"
    filepath = os.path.join(CACHE_FOLDER, filename)

    if not os.path.exists(filepath):
        N = [2**lN for lN in logN]
        M = [2 * n for n in N]

        logN_N_M = {"logN": logN, "N": N, "M": M}

        with open(filepath, "wb") as f:
            pickle.dump(logN_N_M, f)

    with open(filepath, "rb") as f:
        logN_N_M = pickle.load(f)
        logN = logN_N_M["logN"]
        N = logN_N_M["N"]
        M = logN_N_M["M"]

    return logN, N, M


def check_ntt_primality(q: int, M: int):
    # Is this in the KM+1 form?
    NTT_comliance = (q - 1) % M
    # It is compliant, go ahead.
    if NTT_comliance == 0:
        # Now, is q a prime?
        is_prime = MillerRabinPrimalityTest(q)
        if is_prime:
            return True
    return False


def generate_message_primes(mbits=None, how_many=11):
    mbits = [28, 60] if mbits is None else mbits

    filename = "message_special_primes.pkl"
    filepath = os.path.join(CACHE_FOLDER, filename)

    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            mprimes = pickle.load(f)
            # return mprimes
    else:
        logN, N, M = generate_N_M()

        mprimes = {}
        for mb in mbits:
            mprimes[mb] = {}
            for m in M:
                # We want to deal with N, and hide M implicitly.
                N = m // 2
                mprimes[mb][N] = []
                current_query = 2**mb - 1
                q_count = 0

                while True:
                    ok = check_ntt_primality(current_query, m)
                    if ok:
                        mprimes[mb][N].append(current_query)
                        q_count += 1

                        # Have we pulled out how_many primes?
                        if q_count == how_many:
                            break

                    # Move onto the next query.
                    current_query -= 2

        with open(filepath, "wb") as f:
            pickle.dump(mprimes, f)

    return mprimes


def maximum_levels(
    N: int, qbits: int = 40, mbits: int = 60, nksk: int = 2
) -> int:
    extra_bits = mbits * (1 + nksk)
    f_levels = (maximum_qbits(N) - extra_bits) / qbits
    return math.floor(f_levels)


def find_the_next_prime(start: int, m: int, up=True) -> int:
    step: int = 2 if up else -2
    current_query: int = start
    while True:
        ok: bool = check_ntt_primality(q=current_query, M=m)
        if ok:
            break
        current_query += step
    return current_query


def generate_alternating_prime_sequence(
    sb: int = 40,
    N: int = 2**15,
    how_many: int = 60,
    optimize: bool = True,
    alternate_directions: bool = True,
    fixed_direction: bool = False,
) -> list:
    m: int = N * 2
    scale: int = 2**sb

    s_primes: list = []

    up: int = scale + 1
    down: int = scale - 1

    if alternate_directions:
        # Initial search.
        up0: int = find_the_next_prime(start=up, m=m)
        down0: int = find_the_next_prime(start=down, m=m, up=False)

        # Initial error.
        eup: int = up0 - scale
        edown: int = scale - down0

        # Set the current direction.
        # True is up
        # This is the NEXT direction. That means if the first item was up, then this direction is down.
        current_direction: bool = False if eup < edown else True

        # Initialize count.
        q_count: int = 0

        cumulative_scale: int = 1

        # The loop.
        while True:
            # Search in the given direction.
            current_query: int = up if current_direction else down
            next_prime: int = find_the_next_prime(
                start=current_query, m=m, up=current_direction
            )
            # cumulative scale progresses as (beta_{i-1} * beta_i)**2

            # cumulative_scale *= scale / next_prime

            # Use Pre-rescale quadratic deviation rule.
            current_dev = scale / next_prime
            cumulative_scale = cumulative_scale**2 * current_dev**2

            # Set the next variable.
            if current_direction:
                up: int = next_prime + 2
                if optimize:
                    searched: int = int((cumulative_scale * scale) // 2 * 2 - 1)
                    down: int = min(down, searched)
            else:
                down: int = next_prime - 2
                if optimize:
                    searched: int = int((cumulative_scale * scale) // 2 * 2 + 1)
                    up: int = max(up, searched)

            # Switch the direction.
            current_direction: bool = not current_direction

            # Store.
            s_primes.append(next_prime)
            q_count += 1

            # Escape.
            if q_count >= how_many:
                break

    else:
        q_count: int = 0
        current_query: int = up if fixed_direction else down
        step: int = 2 if fixed_direction else -2
        while True:
            current_query: int = find_the_next_prime(
                start=current_query, m=m, up=fixed_direction
            )
            s_primes.append(current_query)
            # Move on.
            q_count += 1
            current_query += step

            # Escape.
            if q_count >= how_many:
                break
    return s_primes


def cum_prod(x: list) -> list:
    ret: list = [1]
    for i in range(len(x)):
        ret.append(ret[-1] * x[i])
    return ret[1:]


def pgen_pseq(sb, N, how_many: int) -> list | str:
    # TODO return 정리
    if how_many < 2:
        return f"ERROR!!! sb = {sb}, N = {N}. Not enough primes."

    try:
        res: list = generate_alternating_prime_sequence(
            sb=sb, N=N, how_many=how_many
        )
    except Exception as e:
        # Try with the half how_many.
        res: list = pgen_pseq(sb=sb, N=N, how_many=how_many // 2)
    return res


def generate_scale_primes(how_many=64, ncpu_cutdown=32, verbose=0):
    savefile = os.path.join(CACHE_FOLDER, "scale_primes.pkl")
    if os.path.exists(savefile):
        with open(savefile, "rb") as f:
            result_dict = pickle.load(f)
            return result_dict

    ncpu = multiprocessing.cpu_count()

    # Cut down the number of n-cpus. It tends to slow down after 32.
    ncpu = min(ncpu, ncpu_cutdown)

    logN, N, M = generate_N_M()

    # Scale.
    logS = list(range(20, 55, 5))

    # Generate input packages.
    inputs = []
    for log_n, n in zip(logN, N):
        how_many = 64 if log_n < 16 else 128
        for sb in logS:
            inputs.append((sb, n, how_many))

    logger.info(f"Generating {len(inputs)} primes with {ncpu} cpus...")
    result = Parallel(n_jobs=ncpu, verbose=verbose)(
        delayed(pgen_pseq)(*inp) for inp in inputs
    )

    result_dict = {(sb, N): pr for (sb, N, how_many), pr in zip(inputs, result)}

    with open(savefile, "wb") as f:
        pickle.dump(result_dict, f)

    return result_dict


def measure_scale_primes_quality(sb: int = 40, N: int = 2**15):
    scale_primes: dict = generate_scale_primes()

    p: list = scale_primes[(sb, N)]

    # Check every prime in the sequence is unique.
    unique_p = sorted(set(p))
    assert len(unique_p) == len(
        p
    ), "There are repeating primes in the generate primes set!!!"

    scale = 2**sb
    e = [scale / pi for pi in p]

    y = cum_prod(e)

    plt.plot(y, label=f"Scale bits={sb}, logN={math.log2(N)}")

    # How many primes?
    print(f"I have {len(p)} primes in the set.")

    # Error propagation.
    q = np.array(y) - 1
    print(f"Max. relative error is {np.abs(q).max():.3e}.")
    print(f"Min. relative error is {np.abs(q).min():.3e}.")
    print(f"Error expanded {np.abs(q).max() / np.abs(q)[0]:.3f} times.")
