from tiberate import CkksEngine

engine = CkksEngine(rng_class="csprng")

engine.encodecrypt([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
