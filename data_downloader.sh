#!/bin/bash

mkdir data && mkdir data/deepsig
cd data/deepsig

wget -O "RML2016.10a.tar.bz2" "http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2?__hstc=24938661.58417cade1c8eea7e2454a05c27d5ecd.1597090293538.1605204807442.1605644485703.36&__hssc=24938661.2.1605644485703&__hsfp=2019803611"

wget -O "RML2016.10b.tar.bz2" "http://opendata.deepsig.io/datasets/2016.10/RML2016.10b.tar.bz2?__hstc=24938661.58417cade1c8eea7e2454a05c27d5ecd.1597090293538.1605204807442.1605644485703.36&__hssc=24938661.2.1605644485703&__hsfp=2019803611"

wget -O "2016.04C.multisnr.tar.bz2" "http://opendata.deepsig.io/datasets/2016.04/2016.04C.multisnr.tar.bz2?__hstc=24938661.58417cade1c8eea7e2454a05c27d5ecd.1597090293538.1605204807442.1605644485703.36&__hssc=24938661.2.1605644485703&__hsfp=2019803611"


tar -xvf ./RML2016.10a.tar.bz2
tar -xvf ./RML2016.10b.tar.bz2
tar -xvf ./2016.04C.multisnr.tar.bz2

rm *bz2

