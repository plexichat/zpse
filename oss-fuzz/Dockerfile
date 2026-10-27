FROM gcr.io/oss-fuzz-base/base-builder-clang
RUN apt-get update && apt-get install -y make
WORKDIR /src
COPY . /src/zpse
WORKDIR /src/zpse
