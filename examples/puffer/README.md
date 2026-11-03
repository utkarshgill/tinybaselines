# PufferLib Squared Continuous

Native CPU smoke test on macOS. It calls PufferLib's reset and step functions
directly and opens no window. Run from the tinybaselines checkout:

```sh
puffer_smoke=$(mktemp -d)
git init "$puffer_smoke/pufferlib"
git -C "$puffer_smoke/pufferlib" fetch --depth=1 https://github.com/PufferAI/PufferLib.git ea7f989371270d2948f96728bda8aec5e7634a4c
git -C "$puffer_smoke/pufferlib" checkout --detach FETCH_HEAD
curl -fL https://github.com/raysan5/raylib/releases/download/5.5/raylib-5.5_macos.tar.gz \
  -o "$puffer_smoke/raylib.tar.gz"
tar -xzf "$puffer_smoke/raylib.tar.gz" -C "$puffer_smoke"
clang -O2 -I"$puffer_smoke/pufferlib" -I"$puffer_smoke/pufferlib/src" \
  -I"$puffer_smoke/raylib-5.5_macos/include" examples/puffer/squared_smoke.c \
  "$puffer_smoke/raylib-5.5_macos/lib/libraylib.a" \
  -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL \
  -o "$puffer_smoke/squared_smoke"
"$puffer_smoke/squared_smoke"
```
