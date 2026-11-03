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

With tinybaselines installed, build the Python adapter's shared library and run
its reset checks and one PPO update:

```sh
clang -O2 -dynamiclib -I"$puffer_smoke/pufferlib" -I"$puffer_smoke/pufferlib/src" \
  -I"$puffer_smoke/raylib-5.5_macos/include" examples/puffer/squared_bridge.c \
  "$puffer_smoke/raylib-5.5_macos/lib/libraylib.a" \
  -framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL \
  -o "$puffer_smoke/libsquared.dylib"
DEV=CPU BEAM=0 python -m examples.puffer.smoke "$puffer_smoke/libsquared.dylib"
```

Train for 204,800 transitions across four environments:

```sh
DEV=CPU BEAM=0 python -m examples.puffer.train "$puffer_smoke/libsquared.dylib" \
  --output runs/puffer-squared-ppo
```

The run saves `ppo.safetensors` and records goals over 100 sampled episodes
before training and after checkpoint reload.

Add `--observations relative` to use the agent's centered row and column and the
signed row and column offset to the target, all divided by 5. Use the same
observation mode when evaluating a saved policy.
