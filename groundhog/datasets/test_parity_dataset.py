from NParity_dataset  import NParityIterator

batch_size = 1
stop = 4
start = 0
path = "/data/lisa/exp/caglargul/codes/python/nbit_parity_data/par_fil_npar_2_nsamp_4_det.npy"

parity_dataset = NParityIterator(batch_size,
                                 stop,
                                 start=start,
                                 max_iters=4,
                                 use_hints=True,
                                 path=path)

for data in parity_dataset:
    print(data)
