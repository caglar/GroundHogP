from Pentomino_dataset  import PentominoIterator

batch_size = 1
stop = 10
start = 0

parity_dataset = PentominoIterator(batch_size=batch_size,
                                   stop=stop,
                                   start=start,
                                   use_infinite_loop=False,
                                   names="pento64x64_40k_64patches_seed_975168712_64patches.npy")
y_sum = 0
for data in parity_dataset:
    y_sum += data[1][0]
    print(data)
print(y_sum)
