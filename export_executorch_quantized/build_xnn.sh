rm -rf cmake-out && mkdir cmake-out

cmake ../executorch -DCMAKE_INSTALL_PREFIX=cmake-out \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DPYTHON_EXECUTABLE=python \
  -Bcmake-out

cmake --build cmake-out -j9 --target install --config Release