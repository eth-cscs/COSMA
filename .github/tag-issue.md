---
title: cmake project version {{ env.CMAKE_VERSION }} does not match git tag {{ env.GIT_VERSION }}
labels: bug
---
The cmake version should be in sync with the git version to ensure the correct file names and sonames of shared libraries.