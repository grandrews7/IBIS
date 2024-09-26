FROM python:3.6
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"


RUN apt-get update && \
    apt-get install -y \
    build-essential \
    libxml2-dev libcurl4-openssl-dev libssl-dev libv8-dev \
    imagemagick libxml-simple-perl libxml-sax-expat-perl \
    libconfig-json-perl  libhtml-treebuilder-libxml-perl libhtml-template-perl \
    libhtml-parser-perl zlib1g-dev libxslt-dev \
    libcairo2-dev libxt-dev \
    bedtools \
    wget \
    git \
    r-base r-base-dev \
    libopenblas-dev \
    libcurl4 libcurl4-openssl-dev \
    libmagick++-dev \
    libmpfr-dev \
    libgmp3-dev \
    libudunits2-dev libharfbuzz-dev libfribidi-dev \
    libglpk-dev
    
    
    
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir \
    scikit-learn \
    pyfaidx \
    pyBigWig \
    jupyterlab \
    scipy \
    matplotlib \
    umap-learn \
    logomaker \
    pysam \
    pandas \
    tqdm \
    deeptools \
    biopython \
    bioframe \
    tensorflow==1.15.0

RUN pip install h5py==2.10.0 --force-reinstall

RUN mkdir /opt/meme
ADD http://meme-suite.org/meme-software/5.4.1/meme-5.4.1.tar.gz /opt/meme
WORKDIR /opt/meme/
RUN tar zxvf meme-5.4.1.tar.gz && rm -fv meme-5.4.1.tar.gz
RUN cd /opt/meme/meme-5.4.1 && \
    ./configure --prefix=/opt  --enable-build-libxml2 --enable-build-libxslt  && \
    make && \
    make install && \
    rm -rfv /opt/meme

ENV PATH="/opt/libexec/meme-5.4.1:/opt/bin:${PATH}"

CMD ["/bin/bash"]