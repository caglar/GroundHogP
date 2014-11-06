#!/bin/bash 

python preprocess.py -d /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/vocab.en.pkl -v 30000 -b /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/iwslt.binarized_text.en.pkl -p /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/IWSLT14.TED.tst2013.en-tr.en.tok.txt
python invert-dict.py /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/vocab.en.pkl /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/ivocab.en.pkl 

python convert-pkl2hdf5.py /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/iwslt.binarized_text.en.pkl /data/lisatmp2/gulcehrc/iwslt/test_data/en-tr/iwslt.binarized_text.en.h5

python shuffle-hdf5.py /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.h5 /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.h5\
    /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.shuff.h5 /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.shuff.h5
