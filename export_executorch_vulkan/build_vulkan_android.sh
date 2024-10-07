# r25c is supposedly recommended but didnt work for me
# export ANDROID_NDK=/home/kindersc/Android/android-ndk-r25c-linux/android-ndk-r25c
export ANDROID_NDK=/home/kindersc/Android/android-ndk-r27b-linux/android-ndk-r27b

export ANDROID_ABI=x86_64
# can also use for arm64 devices
# export ANDROID_ABI=arm64-v8a

rm -rf cmake-android-out && mkdir cmake-android-out

cmake ../executorch -DCMAKE_INSTALL_PREFIX=cmake-android-out \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=$ANDROID_ABI \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_BUILD_VULKAN=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DEXECUTORCH_BUILD_KERNELS_QUANTIZED=ON \
  -DPYTHON_EXECUTABLE=python \
  -Bcmake-android-out

cmake --build cmake-android-out -j16

cmake --build cmake-android-out --target vulkan_executor_runner -j32
