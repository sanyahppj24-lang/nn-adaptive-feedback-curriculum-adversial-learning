import torch
from model import Alice, Bob, Eve

BATCH_SIZE = 64
MSG_LEN = 16
KEY_LEN = 16

alice = Alice(MSG_LEN, KEY_LEN)
bob = Bob(MSG_LEN, KEY_LEN)
eve = Eve(MSG_LEN, KEY_LEN)

message = torch.randint(
    0, 2,
    (BATCH_SIZE, MSG_LEN)
).float()

message = message * 2 - 1

key = torch.randint(
    0, 2,
    (BATCH_SIZE, KEY_LEN)
).float()

key = key * 2 - 1

cipher = alice(message, key)

bob_out = bob(cipher, key)

eve_out = eve(cipher)

print(cipher.shape)
print(bob_out.shape)
print(eve_out.shape)