# from vdtoys.plot import diff_distribution
import torch

from tiberate import CkksEngine
from tiberate.typing import Ciphertext, Plaintext

# Engine creation
engine = CkksEngine()
print(engine)
# Some dummy data
data = torch.randn(8192)
# Encrypt some data
ct = engine.encodecrypt(data)
# Some plaintext with cache
pt = Plaintext(data)
# Save and load ciphertext
ct.save("./ct.pkl")
ct = Ciphertext.load("./ct.pkl")
# Operations with plaintext
ct = engine.pc_mult(pt, ct)  # Multiplication
ct = engine.pc_add(pt, ct)  # Addition
print(pt)  # Print the plaintext information
# Ciphertext operations
ct = engine.cc_mult(ct, ct)  # Multiplication
ct = engine.cc_add(ct, ct)  # Addition
ct = engine.rotate_single(ct, engine.rotk[1])  # Rotation
ct = engine.rotate_single(ct, engine.rotk[-1])  # Rotate back
# Decryption
whatever = engine.decryptcode(ct, is_real=True)
# Error distribution
data = data * data + data
data *= data
data += data
diff = data - whatever[:8192]
print(f"Mean: {diff.mean()}, Std: {diff.std()}")
# plt = diff_distribution(diff)
# plt.show()
