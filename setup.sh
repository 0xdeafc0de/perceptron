#!/usr/bin/env bash
# setup.sh — Download MNIST dataset and build all models
#
# Usage:
#   ./setup.sh           # download data + build all
#   ./setup.sh --data    # download data only
#   ./setup.sh --build   # build only (skip download)
#   ./setup.sh --test    # run a quick smoke test after building

set -euo pipefail

MIRROR="https://ossci-datasets.s3.amazonaws.com/mnist"
FILES=(
    "train-images-idx3-ubyte.gz"
    "train-labels-idx1-ubyte.gz"
    "t10k-images-idx3-ubyte.gz"
    "t10k-labels-idx1-ubyte.gz"
)
TRAIN_CSV="mnist_train.csv"
TEST_CSV="mnist_test.csv"

# ── helpers ────────────────────────────────────────────────────────────────────
log()  { printf '\033[1;32m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m  %s\n' "$*"; }
die()  { printf '\033[1;31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

require() {
    command -v "$1" &>/dev/null || die "'$1' is required but not found. Please install it."
}

# ── download ───────────────────────────────────────────────────────────────────
download_data() {
    require python3

    log "Downloading MNIST binary files from $MIRROR..."
    for f in "${FILES[@]}"; do
        if [[ -f "$f" ]]; then
            log "  $f already exists, skipping."
        else
            log "  Downloading $f..."
            python3 -c "
import urllib.request
urllib.request.urlretrieve('$MIRROR/$f', '$f')
print('  Done.')
"
        fi
    done

    log "Converting to CSV..."
    python3 - <<'PYEOF'
import gzip, struct, csv, array, os, sys

def read_images(path):
    with gzip.open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
        data = array.array('B', f.read())
    return n, rows * cols, data

def read_labels(path):
    with gzip.open(path, 'rb') as f:
        struct.unpack('>II', f.read(8))
        data = array.array('B', f.read())
    return data

def convert(img_file, lbl_file, out_csv):
    if os.path.exists(out_csv):
        print(f'  {out_csv} already exists, skipping.')
        return
    print(f'  Converting {img_file} -> {out_csv}...')
    n, pixels, images = read_images(img_file)
    labels = read_labels(lbl_file)
    with open(out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        for i in range(n):
            w.writerow([labels[i]] + list(images[i*pixels:(i+1)*pixels]))
    print(f'  Done: {n} samples written to {out_csv}.')

convert('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 'mnist_train.csv')
convert('t10k-images-idx3-ubyte.gz',  't10k-labels-idx1-ubyte.gz',  'mnist_test.csv')
PYEOF

    log "Dataset ready: $TRAIN_CSV (60000 samples), $TEST_CSV (10000 samples)"
}

# ── build ──────────────────────────────────────────────────────────────────────
build() {
    require gcc

    log "Building binaries..."
    gcc -Wall -O2 -o slp single-layer-perceptron.c -lm
    log "  Built: slp"
    gcc -Wall -O2 -o mlp multi-layer-perceptron.c -lm
    log "  Built: mlp"
    gcc -Wall -O2 -o mini_model mini_model.c -lm
    log "  Built: mini_model"
}

# ── smoke test ─────────────────────────────────────────────────────────────────
run_tests() {
    log "──────────────────────────────────────────"
    log "Test 1: single-layer perceptron (SLP)"
    log "──────────────────────────────────────────"
    ./slp

    log "──────────────────────────────────────────"
    log "Test 2: multi-layer perceptron — XOR (MLP)"
    log "──────────────────────────────────────────"
    ./mlp

    log "──────────────────────────────────────────"
    log "Test 3: mini_model — single sample prediction"
    log "──────────────────────────────────────────"
    if [[ ! -f model.bin ]]; then
        warn "model.bin not found. Run training first: ./mini_model"
        warn "Skipping mini_model prediction test."
    else
        ./mini_model test "$TEST_CSV" 0
        ./mini_model test "$TEST_CSV" 42
    fi

    log "──────────────────────────────────────────"
    log "Test 4: mini_model — batch accuracy on test set (10000 samples)"
    log "──────────────────────────────────────────"
    if [[ ! -f model.bin ]]; then
        warn "model.bin not found. Skipping accuracy test."
    elif [[ ! -f "$TEST_CSV" ]]; then
        warn "$TEST_CSV not found. Run './setup.sh --data' first."
    else
        ./mini_model eval "$TEST_CSV"
    fi
}

# ── training quickstart ────────────────────────────────────────────────────────
train_mini_model() {
    if [[ ! -f "$TRAIN_CSV" ]]; then
        die "$TRAIN_CSV not found. Run './setup.sh --data' first."
    fi
    log "Training mini_model on $TRAIN_CSV (this may take a few minutes)..."
    ./mini_model
}

# ── main ───────────────────────────────────────────────────────────────────────
MODE="${1:-all}"

case "$MODE" in
    --data)   download_data ;;
    --build)  build ;;
    --test)   run_tests ;;
    --train)  train_mini_model ;;
    all)      download_data; build ;;
    *)
        echo "Usage: $0 [--data | --build | --train | --test | all]"
        echo ""
        echo "  (no args / all)  Download data and build all binaries"
        echo "  --data           Download and convert MNIST CSVs only"
        echo "  --build          Build all binaries only"
        echo "  --train          Train mini_model on mnist_train.csv"
        echo "  --test           Run smoke tests on all built models"
        exit 1
        ;;
esac

log "Done."
