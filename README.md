# mtl-disfluency-detection

This is a multi-task learning framework to enable the training of one universal incrementaldialogue processing model with four tasks of disfluency detection, language modelling, part-of-speech tagging and utterance segmentation in a simple deep recurrent setting. 

The main model is described here:

Morteza Rohanian and Julian Hough. Re-framing Incremental Deep Language Models for Dialogue Processing with Multi-task Learning. to appear at COLING 2020, Barcelona, December 2020

The basis of these models is the experments in:

Julian Hough and David Schlangen. Joint, Incremental Disfluency Detection and Utterance Segmentation from Speech. In Proceedings of EACL 2017, Valencia, April 2017.
[https://github.com/clp-research/deep_disfluency]

The CRF implementation is based on:

Marek Rei. Semi-supervised Multitask Learning for Sequence Labeling. In Proceedings ACL 2017, Vancouver, August 2017.


The MTL loss implementation is based on:

Alex Kendall, Yarin Gal, and Roberto Cipolla. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition 2018,  Salt Lake City, June 2018.
