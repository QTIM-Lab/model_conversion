rm -rf cmake-out && mkdir cmake-out

cmake ../executorch -DCMAKE_INSTALL_PREFIX=cmake-out \
  -Bcmake-out

cmake --build cmake-out --target executor_runner -j9