export ANDROID_NDK=/home/kindersc/Android/android-ndk-r27b-linux/android-ndk-r27b
# export ANDROID_ABI=arm64-v8a
# I assume you use x86_64 but if on arm64 please switch back
export ANDROID_ABI=x86_64

rm -rf cmake-android-out && mkdir cmake-android-out

# Build the core executorch library
cmake ../executorch -DCMAKE_INSTALL_PREFIX=cmake-android-out \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
  -Bcmake-android-out

cmake --build cmake-android-out -j16 --target install

# Build the android extension
# Path may be different...
cmake extension/android \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}"/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI="${ANDROID_ABI}" \
  -DCMAKE_INSTALL_PREFIX=cmake-android-out \
  -Bcmake-android-out/extension/android

cmake --build cmake-android-out/extension/android -j16