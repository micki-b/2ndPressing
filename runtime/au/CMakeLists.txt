# Audio Unit (AUv2) wrapper for the 2ndPressing CLAP plugin, via clap-wrapper.
#
# The .component is a thin wrapper that loads "2ndPressing Declicker.clap"
# at runtime from the standard CLAP locations (~/Library/Audio/Plug-Ins/CLAP
# or /Library/Audio/Plug-Ins/CLAP). Build the CLAP first:
#
#   cd runtime && cargo xtask bundle pressing-plugin --release
#   cmake -S au -B au/build -DCMAKE_BUILD_TYPE=Release && cmake --build au/build
#
# Install: copy the .clap to ~/Library/Audio/Plug-Ins/CLAP and the .component
# to ~/Library/Audio/Plug-Ins/Components, then validate with
#   auval -v aufx Dclk SnPr

cmake_minimum_required(VERSION 3.21)
set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0" CACHE STRING "macOS deployment target")
project(pressing_auv2 VERSION 0.1.0 LANGUAGES C CXX OBJC OBJCXX)

# clap-wrapper only sets the C++ standard when it is the top-level project;
# its AudioUnitSDK target needs C++17 (std::byte, lock_guard CTAD).
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 17)

include(FetchContent)
set(CLAP_WRAPPER_DOWNLOAD_DEPENDENCIES TRUE CACHE BOOL "" FORCE)
set(CLAP_WRAPPER_BUILD_AUV2 TRUE CACHE BOOL "" FORCE)
FetchContent_Declare(clap-wrapper
    GIT_REPOSITORY https://github.com/free-audio/clap-wrapper.git
    GIT_TAG 806bba0   # 0.17.0 release
)
FetchContent_MakeAvailable(clap-wrapper)

set(PLUGIN_NAME "2ndPressing Declicker")   # must match the .clap bundle name

add_library(pressing_auv2 MODULE)
target_add_auv2_wrapper(
    TARGET pressing_auv2
    OUTPUT_NAME "${PLUGIN_NAME}"
    BUNDLE_IDENTIFIER "org.2ndpressing.declicker.auv2"
    BUNDLE_VERSION "${PROJECT_VERSION}"
    INSTRUMENT_TYPE "aufx"          # effect
    MANUFACTURER_NAME "2ndPressing"
    MANUFACTURER_CODE "SnPr"
    SUBTYPE_CODE "Dclk"
)
