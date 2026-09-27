#!/bin/sh
# Build dependencies for the dist package on a RedHat-compatible host
# (RHEL / Rocky / Alma / Fedora): git, make, cmake, and a compiler that
# can build the pinned llama.cpp. RHEL 8's system GCC 8.5 is too old, so
# a gcc-toolset (>= 9) is installed there; the Makefile activates it by
# itself at build time — no need to source anything manually.
set -u

SUDO=sudo
[ "$(id -u)" = 0 ] && SUDO=

if command -v dnf >/dev/null 2>&1; then PKG=dnf
elif command -v yum >/dev/null 2>&1; then PKG=yum
else
  echo "error: need dnf or yum (RedHat-compatible host)" >&2
  exit 1
fi

echo "==> git, make, cmake"
$SUDO $PKG install -y git make cmake

major=$(gcc -dumpversion 2>/dev/null | cut -d. -f1)
if [ -n "$major" ] && [ "$major" -ge 9 ]; then
  echo "==> system GCC $major can build the pinned llama.cpp"
else
  echo "==> system GCC ${major:-missing} is too old — installing a gcc-toolset"
  ok=
  for v in 14 13 12 11 10 9; do
    if $SUDO $PKG install -y gcc-toolset-$v-gcc gcc-toolset-$v-gcc-c++; then
      ok=/opt/rh/gcc-toolset-$v/enable
      break
    fi
  done
  [ -n "$ok" ] || { echo "error: no gcc-toolset >= 9 could be installed" >&2; exit 1; }
  echo "==> installed $ok (the Makefile sources it itself at build time)"
fi

echo "==> done — now run: make"
