#!/bin/bash 

python preprocess.py -d /data/lisatmp2/gulcehrc/iwslt/vocab.tr.pkl --char -v 255 -b /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.pkl -p /data/lisatmp2/gulcehrc/iwslt/train.tok.tr.txt
python preprocess.py -d /data/lisatmp2/gulcehrc/iwslt/vocab.en.pkl -v 30000 -b /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.pkl -p /data/lisatmp2/gulcehrc/iwslt/train.tok.en.txt


python convert-pkl2hdf5.py /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.pkl /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.h5
python convert-pkl2hdf5.py /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.pkl /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.h5


python shuffle-hdf5.py /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.h5 /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.h5\
    /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.shuff.h5 /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.shuff.h5 

python shuffle-hdf5.py /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.h5 /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.h5\
    /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.en.shuff.h5 /data/lisatmp2/gulcehrc/iwslt/iwslt.binarized_text.tr.shuff.h5
